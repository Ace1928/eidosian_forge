import itertools
import random
from typing import Tuple
from .. import language as tl
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
from .tl_lang import (TritonLangProxy, WrappedTensor, _primitive_to_tensor,
class GridSelector:
    """
    Entry point of the debugger
    """

    def __init__(self, func):
        version = torch.__version__
        assert version[0] == '2', f'Triton Debugger only supports torch >= 2.0, using {version}'
        self.func = func

    def __getitem__(self, grid):
        return DebuggerFunction(self.func, grid)

    def __call__(self, *args, **kwargs):
        return DebuggerFunction(self.func)(*args, **kwargs)