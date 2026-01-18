import functools
from typing import Dict, Optional, Tuple, Union
import torch
from torch.cuda import _CudaDeviceProperties
def _device(device: Optional[Union[torch.device, int]]) -> int:
    if device is not None:
        if isinstance(device, torch.device):
            assert device.type == 'cuda'
            device = device.index
        return device
    return current_device()