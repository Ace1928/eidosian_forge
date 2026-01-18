import contextlib
import warnings
from typing import Dict, List, Optional
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._tensor.placement_types import DTensorSpec, Shard
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
@contextlib.contextmanager
def _distribute_region(self, spec: DTensorSpec):
    if not self.rng_state_is_sync('tensor-parallel-rng'):
        raise RuntimeError('TensorParallelRNGTracker requires the random state to be synchronized before entering into a distribute region!')
    if self.distribute_region_enabled:
        with torch.random.fork_rng(self._devices, device_type=self._device_type):
            self._device_handle.set_rng_state(self.rng_states['tensor-parallel-rng'])
            try:
                yield
            finally:
                self.rng_states['tensor-parallel-rng'] = self._device_handle.get_rng_state()
    else:
        yield