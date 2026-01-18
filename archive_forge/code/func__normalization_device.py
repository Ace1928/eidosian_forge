import torch
from torch._C import _rename_privateuse1_backend, _get_privateuse1_backend_name
from typing import List, Optional, Union
def _normalization_device(custom_backend_name: str, device: Optional[Union[int, str, torch.device]]=None) -> int:

    def _get_current_device_index():
        _get_device_index = 'current_device'
        if hasattr(torch, custom_backend_name) and hasattr(getattr(torch, custom_backend_name), _get_device_index):
            return getattr(getattr(torch, custom_backend_name), _get_device_index)()
        else:
            return 0
    if device is None:
        return _get_current_device_index()
    elif isinstance(device, str):
        device = torch.device(device)
    if isinstance(device, torch.device):
        if device.type != custom_backend_name:
            raise RuntimeError(f'Invalid device, must be {custom_backend_name} device')
        elif device.index is None:
            device_idx = _get_current_device_index()
        else:
            device_idx = device.index
    else:
        device_idx = device
    return device_idx