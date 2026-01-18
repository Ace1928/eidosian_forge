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
def _get_current_function_mode_stack():
    stack_len = _len_torch_function_stack()
    return [_get_function_stack_at(i) for i in range(stack_len)]