import torch.distributed as dist
from torch._C._distributed_c10d import (
from torch.futures import Future
from typing import List
from torch import Tensor
def _allgather_base(self, output_tensor, input_tensor, opts=AllgatherOptions()):
    chunks = output_tensor.chunk(self._world_size)
    for chunk in chunks:
        chunk.copy_(input_tensor)
    return ret_work(output_tensor)