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
class MockTensor:
    """
    Can be used in place of real tensors when calling:
        kernel.warmup(MockTensor(torch.float32), ...)
    """

    @staticmethod
    def wrap_dtype(arg):
        if arg.__class__.__name__ == 'dtype' and arg.__module__ == 'torch':
            return MockTensor(arg)
        return arg

    def __init__(self, dtype):
        self.dtype = dtype

    @staticmethod
    def data_ptr():
        return 0