import contextlib
import ctypes
import glob
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, cast
import torch
from .common import BaseOperator, get_operator, get_xformers_operator, register_operator
@classmethod
def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
    if func._overloadpacket not in SPARSE24_DISPATCH_CUSPARSELT:
        raise NotImplementedError(f"{cls.__name__} only supports a specific set of operations, can't perform requested op ({func.__name__})")
    return SPARSE24_DISPATCH_CUSPARSELT[func._overloadpacket](func, types, args, kwargs)