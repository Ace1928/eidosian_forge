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
class BaseTorchFunctionMode(TorchFunctionMode):

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        return func(*args, **kwargs)