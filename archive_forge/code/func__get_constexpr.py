import itertools
import random
from typing import Tuple
from .. import language as tl
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
from .tl_lang import (TritonLangProxy, WrappedTensor, _primitive_to_tensor,
def _get_constexpr(self):
    result = []
    for name, annotation in self.func.__annotations__.items():
        if annotation is lcore.constexpr:
            result.append(name)
    return result