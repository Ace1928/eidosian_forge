import sys
from functools import wraps, partial
import torch
import torch.distributed as dist
from torch.distributed import rpc
from torch.testing._internal.common_distributed import (
class ShardedTensorTestBase(MultiProcessTestCase):

    @property
    def world_size(self):
        return TEST_GPU_NUM

    def init_pg(self, backend='nccl'):
        if backend not in ['nccl', 'gloo', 'mpi']:
            raise RuntimeError(f'Backend {backend} not supported!')
        dist.init_process_group(backend=backend, world_size=self.world_size, rank=self.rank, init_method=f'file://{self.file_name}')
        if backend == 'nccl':
            torch.cuda.set_device(self.rank)

    def init_rpc(self):
        rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=tp_transports())
        rpc_backend_options.init_method = f'file://{self.file_name}'
        for rank in range(self.world_size):
            rpc_backend_options.set_device_map(f'worker{rank}', {rank: self.rank, self.rank: rank})
        rpc.init_rpc(name='worker%d' % self.rank, rank=self.rank, world_size=self.world_size, rpc_backend_options=rpc_backend_options)

    def init_comms(self, init_rpc=True, backend='nccl'):
        if init_rpc:
            self.init_rpc()
        self.init_pg(backend=backend)

    def destroy_comms(self, destroy_rpc=True):
        dist.barrier()
        if destroy_rpc:
            rpc.shutdown()
        dist.destroy_process_group()

    def setUp(self) -> None:
        super().setUp()
        self._spawn_processes()

    def assert_sharded_tensor_equal(self, st1, st2):
        st1_local_shards = st1.local_shards()
        st2_local_shards = st2.local_shards()
        self.assertEqual(len(st1_local_shards), len(st2_local_shards))
        for i, st1_local_shard in enumerate(st1_local_shards):
            self.assertEqual(st1_local_shard.tensor, st2_local_shards[i].tensor)
            self.assertEqual(st1_local_shard.metadata, st2_local_shards[i].metadata)
        self.assertEqual(st1.metadata(), st2.metadata())
        self.assertEqual(st1.sharding_spec(), st2.sharding_spec())
        self.assertEqual(len(st1.remote_shards()), len(st2.remote_shards()))