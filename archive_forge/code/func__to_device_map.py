from typing import Dict, List, Optional, Union
import torch
from torch._C._distributed_rpc import _TensorPipeRpcBackendOptionsBase
from . import constants as rpc_contants
def _to_device_map(device_map: Dict[DeviceType, DeviceType]) -> Dict[torch.device, torch.device]:
    full_device_map: Dict[torch.device, torch.device] = {}
    reverse_map: Dict[torch.device, torch.device] = {}
    for k, v in device_map.items():
        k, v = (torch.device(k), torch.device(v))
        if v in reverse_map:
            raise ValueError(f'`device_map` only supports 1-to-1 mapping, trying to map {k} and {reverse_map[v]} to {v}')
        full_device_map[k] = v
        reverse_map[v] = k
    return full_device_map