import logging
import math
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING, Union
import torch
from torch.distributed import is_available
def _get_or_create_default_group(self):
    default_initialized = is_initialized()
    if not default_initialized:
        init_process_group()
    world_size = get_world_size()
    if self.mesh.numel() > world_size:
        raise RuntimeError(f'Mesh should not be bigger than default world size, but found {self.mesh.numel()} ranks!')
    device_handle = _get_device_handle(self.device_type)
    if not default_initialized and device_handle:
        num_devices_per_host = device_handle.device_count()
        if world_size > num_devices_per_host and world_size % num_devices_per_host != 0:
            raise RuntimeError(f'DeviceMesh only support homogeneous hardware, but found {world_size} ranks and {num_devices_per_host} {self.device_type} devices!')
        device_handle.set_device(get_rank() % num_devices_per_host)
    rank_coords = (self.mesh == get_rank()).nonzero()
    assert rank_coords.size(0) in (0, 1)
    self._coordinate_on_dim: Optional[List[int]] = rank_coords[0].tolist() if rank_coords.size(0) > 0 else None
    return _get_default_group()