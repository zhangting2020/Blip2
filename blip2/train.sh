#FLAGS_allocator_strategy='naive_best_fit'
export USE_FLASH_ATTN=True
#export CUDA_VISIBLE_DEVICES=4,5
export CUDA_VISIBLE_DEVICES=4,5,6,7
#nsys profile --stats true -w true -t cuda,nvtx,osrt,cudnn,cublas --capture-range=cudaProfilerApi -x true --force-overwrite true --cuda-memory-usage true -o blip2_bs250_no_flash_attn
python -m paddle.distributed.launch train_paddle.py --cfg-path lavis_paddle/projects/blip2/train/pretrain_stage2.yaml
