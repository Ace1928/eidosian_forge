import torch.distributed as dist
from torch._C._distributed_c10d import (
from torch.futures import Future
from typing import List
from torch import Tensor
def _create_fake_pg(prefix_store, rank, world_size, timeout):
    return FakeProcessGroup(rank, world_size)