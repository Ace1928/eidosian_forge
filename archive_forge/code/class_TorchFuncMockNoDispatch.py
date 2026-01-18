import itertools
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Set, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.profiler
import torch.utils.hooks
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
from torch.utils._pytree import tree_map
from ..ops.common import FUNC_TO_XFORMERS_OPERATOR
from .device_limits import get_device_limits
from .profiler import _Profiler
class TorchFuncMockNoDispatch:
    """
    Wraps a method to call it without the custom
    pytorch dispatcher
    """

    def __init__(self, pt_impl):
        self.pt_impl = pt_impl

    def __get__(self, obj, c):
        return partial(self, obj)

    def __call__(self, obj, *args, **kwargs):
        with _pop_mode_temporarily():
            return self.pt_impl(obj, *args, **kwargs)