import sys
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
from functools import partial, reduce
import torch
import torch.distributed as dist
import weakref
from torch._C._distributed_c10d import (
from torch.distributed.distributed_c10d import _CollOp, _store_based_barrier, P2POp
from torch.futures import Future
from torch.utils import _pytree as pytree
class AllToAll:

    @torch.no_grad()
    def work(self, data):
        world_size = len(data)
        for dest_rank in range(world_size):
            output_tensor_list, _ = data[dest_rank]
            for src_rank in range(world_size):
                _, input_tensor_list = data[src_rank]
                output_tensor_list[src_rank].copy_(input_tensor_list[dest_rank])