import os
import textwrap
from enum import auto, Enum
from traceback import extract_stack, format_exc, format_list, StackSummary
from typing import cast, NoReturn, Optional
import torch._guards
from . import config
from .config import is_fbcode
from .utils import counters
import logging
class BackendCompilerFailed(TorchDynamoException):

    def __init__(self, backend_fn, inner_exception):
        self.backend_name = getattr(backend_fn, '__name__', '?')
        self.inner_exception = inner_exception
        msg = f'backend={self.backend_name!r} raised:\n{type(inner_exception).__name__}: {inner_exception}'
        super().__init__(msg)