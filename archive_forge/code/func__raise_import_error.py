from typing import Optional
import torch
def _raise_import_error(e):
    if torch.cuda.get_device_capability() < (8, 0):
        raise ImportError('punica LoRA kernels require compute capability >= 8.0') from e
    else:
        raise ImportError('punica LoRA kernels could not be imported. If you built vLLM from source, make sure VLLM_INSTALL_PUNICA_KERNELS=1 env var was set.') from e