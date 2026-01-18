import contextlib
import functools
import itertools
import logging
import os
import sys
import traceback
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
from weakref import ReferenceType
import torch
import torch._custom_op
import torch._logging
from torch._guards import Source
from torch._ops import OpOverload
from torch._prims_common import (
from torch._subclasses.meta_utils import MetaConverter
from torch._utils import render_call
from torch.fx.operator_schemas import normalize_function
from torch.multiprocessing.reductions import StorageWeakRef
from torch.overrides import TorchFunctionMode
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import (
from torch.utils._pytree import PyTree, tree_map
from torch.utils._stats import count, count_label
from torch.utils.weak import WeakIdRef
class FakeCopyMode(TorchFunctionMode):

    def __init__(self, fake_mode):
        self.fake_mode = fake_mode

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
        if func == torch._C.TensorBase.clone:
            return func(self.fake_mode.from_tensor(args[0], static_shapes=True), **kwargs)
        elif func == torch.Tensor.__deepcopy__:
            assert len(args) == 2 and len(kwargs) == 0
            tensor, memo = args
            if id(tensor) in memo:
                return memo[id(tensor)]
            out = self.fake_mode.from_tensor(tensor, static_shapes=True)
            memo[id(tensor)] = out
            return out
        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)