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
class AllGather:

    @torch.no_grad()
    def work(self, data):
        for src_rank in range(len(data)):
            in_tensor_list = data[src_rank][1]
            assert len(in_tensor_list) == 1
            src_tensor = in_tensor_list[0]
            for dest in data:
                dest_tensor = dest[0][0][src_rank]
                dest_tensor.copy_(src_tensor)