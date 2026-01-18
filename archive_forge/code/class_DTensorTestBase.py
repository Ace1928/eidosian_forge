import itertools
import sys
from functools import wraps
from typing import (
import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec
from torch.testing._internal.common_distributed import (
from torch.distributed._tensor import (
from torch.distributed._tensor.placement_types import Placement
class DTensorTestBase(MultiProcessTestCase):

    @property
    def world_size(self) -> int:
        return NUM_DEVICES

    @property
    def backend(self) -> str:
        return PG_BACKEND

    def build_device_mesh(self) -> DeviceMesh:
        return DeviceMesh(DEVICE_TYPE, list(range(NUM_DEVICES)))

    def init_pg(self) -> None:
        if 'nccl' in self.backend and torch.cuda.device_count() < self.world_size:
            sys.exit(TEST_SKIPS[f'multi-gpu-{self.world_size}'].exit_code)
        if self.backend not in ['nccl', 'gloo', 'mpi', 'cpu:gloo,cuda:nccl']:
            raise RuntimeError(f'Backend {self.backend} not supported!')
        dist.init_process_group(backend=self.backend, world_size=self.world_size, rank=self.rank, init_method=f'file://{self.file_name}')
        if 'nccl' in self.backend:
            torch.cuda.set_device(self.rank)

    def destroy_pg(self) -> None:
        dist.barrier()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def _test_op(self, mesh: DeviceMesh, op_call, *args, **kwargs) -> None:
        out = op_call(*args, **kwargs)
        dtc = DTensorConverter(mesh, args, kwargs)
        for d_args, d_kwargs in dtc:
            self.assertEqual(dtc.successful(), True)
            d_out = op_call(*d_args, **d_kwargs)
            self.assertEqual(d_out.full_tensor(), out)

    def run_subtests(self, *args, **kwargs):
        return run_subtests(self, *args, **kwargs)