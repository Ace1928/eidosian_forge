import __future__  # noqa: F404
import collections
import functools
import types
import warnings
from typing import Dict, Set, List, Any, Callable, Iterable, Type, Tuple
from functools import wraps
import contextlib
import torch
from torch._C import (
@functools.lru_cache(None)
def _get_tensor_methods() -> Set[Callable]:
    """ Returns a set of the overridable methods on ``torch.Tensor`` """
    overridable_funcs = get_overridable_functions()
    methods = set(overridable_funcs[torch.Tensor])
    return methods