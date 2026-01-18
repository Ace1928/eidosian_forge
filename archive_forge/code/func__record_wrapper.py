import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
def _record_wrapper(self, ptr):
    self._register_tensor_ptr(ptr)
    self.ptr_alias_count.setdefault(ptr, 0)
    self.ptr_alias_count[ptr] += 1
    self.cleanup_count += 1