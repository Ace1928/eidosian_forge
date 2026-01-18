import enum
import functools
import inspect
import io
import logging
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar
import torch
import torch.utils._cuda_trace as cuda_trace
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
class _TensorsAccessed:

    def __init__(self):
        self.accesses: Dict[DataPtr, TensorInfo] = {}

    def ensure_tensor_exists(self, data_ptr: DataPtr) -> None:
        if data_ptr not in self.accesses:
            logger.info('Found tensor with pointer: %s, but no matching tensor allocation in the trace. Backfilling the trace now. Perhaps the sanitizer was enabled after some torch operations?', data_ptr)
            self.create_tensor(data_ptr, None)

    def ensure_tensor_does_not_exist(self, data_ptr: DataPtr) -> None:
        if data_ptr in self.accesses:
            logger.info("Found duplicate tensor allocation in the trace for tensor with pointer: %s. Assuming the trace for tensor deallocation wasn't caught and backfilling it now. Perhaps the sanitizer was enabled after some torch operations?", data_ptr)
            self.delete_tensor(data_ptr)

    def create_tensor(self, data_ptr: DataPtr, stack_trace: Optional[traceback.StackSummary]) -> None:
        self.accesses[data_ptr] = TensorInfo(stack_trace)

    def delete_tensor(self, data_ptr: DataPtr) -> None:
        del self.accesses[data_ptr]

    def were_there_reads_since_last_write(self, data_ptr: DataPtr) -> bool:
        return True if self.accesses[data_ptr].reads else False

    def get_allocation_stack_trace(self, data_ptr: DataPtr) -> Optional[traceback.StackSummary]:
        return self.accesses[data_ptr].allocation_stack_trace

    def get_write(self, data_ptr: DataPtr) -> Optional[Access]:
        return self.accesses[data_ptr].write

    def get_reads(self, data_ptr: DataPtr) -> List[Access]:
        return self.accesses[data_ptr].reads

    def add_read(self, data_ptr: DataPtr, access: Access) -> None:
        self.accesses[data_ptr].reads.append(access)

    def set_write(self, data_ptr: DataPtr, access: Access) -> None:
        self.accesses[data_ptr].write = access
        self.accesses[data_ptr].reads = []