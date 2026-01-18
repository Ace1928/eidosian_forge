import itertools
import random
from typing import Tuple
from .. import language as tl
from ..language import core as lcore
from . import torch_wrapper
from .core import ExecutionContext
from .memory_map import MemoryMap
from .tl_lang import (TritonLangProxy, WrappedTensor, _primitive_to_tensor,
def _assert_constexpr(self, **kwargs):
    constexp = self._get_constexpr()
    missing = [i for i in constexp if i not in kwargs.keys()]
    assert len(missing) == 0, f'You must specify constexpr {missing}'