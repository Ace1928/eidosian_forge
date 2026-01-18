from __future__ import annotations, division
import ast
import functools
import hashlib
import inspect
import os
import textwrap
from collections import defaultdict, namedtuple
from functools import cached_property
from typing import Callable, Generic, Iterable, List, Optional, TypeVar, Union, cast, overload
from .._C.libtriton.triton import TMAInfos
from ..common.backend import get_backend, get_cuda_version_key
from .interpreter import InterpretedFunction
@staticmethod
def _key_of(arg):
    if hasattr(arg, 'dtype'):
        return arg.dtype
    elif isinstance(arg, bool):
        return 'i1'
    elif isinstance(arg, int):
        if -2 ** 31 <= arg and arg <= 2 ** 31 - 1:
            return 'i32'
        elif 2 ** 63 <= arg and arg <= 2 ** 64 - 1:
            return 'u64'
        else:
            return 'i64'
    elif isinstance(arg, float):
        return 'fp32'
    elif arg is None:
        return None
    else:
        raise TypeError(f'Unsupported type {type(arg)} for {arg}')