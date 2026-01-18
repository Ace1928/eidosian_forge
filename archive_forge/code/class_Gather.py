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
class Gather:

    def __init__(self, dst):
        self.dst = dst

    @torch.no_grad()
    def work(self, data):
        assert len(data[self.dst][0]) == 1
        out_tensor_list = data[self.dst][0][0]
        for rank, each_rank_data in enumerate(data):
            src_in_tensor_list = each_rank_data[1]
            assert len(src_in_tensor_list) == 1
            dest_tensor = out_tensor_list[rank]
            dest_tensor.copy_(src_in_tensor_list[0])