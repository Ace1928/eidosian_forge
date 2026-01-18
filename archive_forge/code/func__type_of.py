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
def _type_of(key):
    if key is None:
        return '*i8'
    dtype_str = str(key).split('.')[-1]
    tys = {'bool': 'i1', 'float8e4nv': 'fp8e4nv', 'float8e5': 'fp8e5', 'float8e4b15': 'fp8e4b15', 'float8e4b15x4': 'fp8e4b15x4', 'float8_e4m3fn': 'fp8e4nv', 'float8_e5m2': 'fp8e5', 'float16': 'fp16', 'bfloat16': 'bf16', 'float32': 'fp32', 'float64': 'fp64', 'int8': 'i8', 'int16': 'i16', 'int32': 'i32', 'int64': 'i64', 'uint8': 'u8', 'uint16': 'u16', 'uint32': 'u32', 'uint64': 'u64'}
    for v in list(tys.values()):
        tys[v] = v
    return key if isinstance(key, str) else f'*{tys[dtype_str]}'