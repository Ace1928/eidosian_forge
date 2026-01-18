import torch.distributed as dist
from torch._C._distributed_c10d import (
from torch.futures import Future
from typing import List
from torch import Tensor
class FakeProcessGroup(dist.ProcessGroup):
    """
    A fake process group (not related to FakeTensor) is a process group which
    doesn't actually do any communication, it just hallucinates some
    communication.  You can run a single rank with a fake process group
    without needing multiple processes (simulates per-rank behavior)

    NOTE: This is not a real process group, and it would produce wrong results
    for every collective. It should be used as a convinient tool when playing
    with distributed but don't care about the actual data.
    """

    def __init__(self, rank, world_size):
        super().__init__(rank, world_size)
        self._rank = rank
        self._world_size = world_size

    def allreduce(self, tensor_list, opts=AllreduceOptions()):
        return ret_work(tensor_list)

    def allreduce_coalesced(self, tensor_list, opts=AllreduceOptions()):
        return ret_work(tensor_list)

    def allgather(self, output_tensors, input_tensor, opts=AllgatherOptions()):
        for chunk in output_tensors[0]:
            chunk.copy_(input_tensor[0])
        return ret_work(output_tensors)

    def reduce_scatter(self, output_tensor, scatter_list, opts=ReduceScatterOptions()):
        return ret_work(output_tensor)

    def _allgather_base(self, output_tensor, input_tensor, opts=AllgatherOptions()):
        chunks = output_tensor.chunk(self._world_size)
        for chunk in chunks:
            chunk.copy_(input_tensor)
        return ret_work(output_tensor)

    def _reduce_scatter_base(self, output_tensor, input_tensor, opts=ReduceScatterOptions()):
        return ret_work(output_tensor)

    def barrier(self, opts=BarrierOptions()):
        pass

    def broadcast(self, tensors: List[Tensor], opts=BroadcastOptions()):
        return ret_work(tensors)

    def scatter(self, output_tensors: List[Tensor], input_tensors: List[List[Tensor]], opts=ScatterOptions()):
        return ret_work(output_tensors)

    def alltoall(self, output_tensors: List[Tensor], input_tensors: List[Tensor], opts=AllToAllOptions()):
        return ret_work(output_tensors)

    def alltoall_base(self, output_tensor: Tensor, input_tensor: Tensor, output_split_sizes: List[int], input_split_sizes: List[int], opts=AllToAllOptions()):
        return ret_work(output_tensor)

    def send(self, tensors: List[Tensor], dstRank: int, tag: int):
        return ret_work(None)

    def recv(self, tensors: List[Tensor], srcRank: int, tag: int):
        return ret_work(tensors)

    def getBackendName(self):
        return 'fake'

    def __repr__(self):
        return f'FakePG world_size:{self._world_size} rank:{self._rank}'