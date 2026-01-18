import logging
import warnings
import weakref
import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d
from typing import List, Optional, cast
class _WaitRegistration:

    def __init__(self, work):
        global work_version
        self.work = work
        self.version = work_version
        self.ptrs = set()
        self.ptr_alias_count = {}
        self.cleanup_count = 0
        work_version += 1

    def _register_tensor_ptr(self, data_ptr):
        global data_ptr_to_work
        data_ptr_to_work[data_ptr] = self
        self.ptrs.add(data_ptr)

    def _record_wrapper(self, ptr):
        self._register_tensor_ptr(ptr)
        self.ptr_alias_count.setdefault(ptr, 0)
        self.ptr_alias_count[ptr] += 1
        self.cleanup_count += 1

    def wait(self):
        if self.work is not None:
            self.work.wait()
            self.work = None
        self.cleanup()

    def decrement_live_tensor(self, ptr):
        self.cleanup_count -= 1
        if self.cleanup_count == 0:
            self.wait()
        else:
            self.ptr_alias_count[ptr] -= 1
            if self.ptr_alias_count[ptr] < 1 and data_ptr_to_work.get(ptr, None) == self:
                del data_ptr_to_work[ptr]

    def cleanup(self):
        for ptr in self.ptrs:
            if data_ptr_to_work.get(ptr, None) == self:
                del data_ptr_to_work[ptr]