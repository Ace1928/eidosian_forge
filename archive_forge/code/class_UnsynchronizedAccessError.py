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
class UnsynchronizedAccessError(SynchronizationError):
    """Stores information about two unsynchronized accesses to one data pointer."""

    def __init__(self, data_ptr: DataPtr, allocation_stack_trace: Optional[traceback.StackSummary], current_access: Access, previous_access: Access):
        self.data_ptr = data_ptr
        self.allocation_stack_trace = allocation_stack_trace
        self.current_access = current_access
        self.previous_access = previous_access

    def __str__(self):

        def format_access(access: Access):
            message.write(f'{access.operator}\n{access.type}')
            if access.aliases:
                message.write(' argument(s) ' + ', '.join(access.aliases))
                if access.is_output:
                    message.write(', and to')
            if access.is_output:
                message.write(' the output')
            message.write(f'\nWith stack trace:\n{''.join(access.stack_trace.format())}\n')
        with io.StringIO() as message:
            message.write(textwrap.dedent(f'                    ============================\n                    CSAN detected a possible data race on tensor with data pointer {self.data_ptr}\n                    Access by stream {self.current_access.stream} during kernel:\n                    '))
            format_access(self.current_access)
            message.write(f'Previous access by stream {self.previous_access.stream} during kernel:\n')
            format_access(self.previous_access)
            if self.allocation_stack_trace:
                message.write(f'Tensor was allocated with stack trace:\n{''.join(self.allocation_stack_trace.format())}')
            else:
                message.write('Trace for tensor allocation not found.')
            return message.getvalue()