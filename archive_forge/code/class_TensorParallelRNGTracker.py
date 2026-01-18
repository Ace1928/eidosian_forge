import contextlib
import warnings
from typing import Dict, List, Optional
import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed._tensor.placement_types import DTensorSpec, Shard
from torch.distributed.device_mesh import _get_device_handle, DeviceMesh
class TensorParallelRNGTracker(RNGStateTracker):

    def __init__(self, device_type: str='cuda'):
        super().__init__(device_type)
        self.rng_states['tensor-parallel-rng'] = self._device_handle.get_rng_state()

    def _manual_seed(self, device_mesh: DeviceMesh, base_seed: int=1234, tp_dim: int=0):
        coordinate = device_mesh.get_coordinate()
        assert coordinate is not None
        tensor_parallel_rank = coordinate[tp_dim]
        MegatronMagicNum = 2718
        tensor_parallel_seed = base_seed + MegatronMagicNum + tensor_parallel_rank
        self.set_seed('tensor-parallel-rng', tensor_parallel_seed)

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