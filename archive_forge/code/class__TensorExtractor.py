import ast
import builtins
import collections
import contextlib
import enum
import inspect
import io
import pickle
import sys
import threading
import types
import typing
import warnings
import weakref
from textwrap import dedent
from typing import (  # noqa: F401
import torch
import torch.distributed.rpc
import torch.package._mangling as package_mangling
from torch._awaits import _Await
from torch._C import _Await as CAwait, Future as CFuture
from torch._sources import fake_range, get_source_lines_and_file, parse_def
from torch.futures import Future
class _TensorExtractor(pickle.Pickler):

    def __init__(self, *args, tensors: List[torch.Tensor], **kwargs):
        super().__init__(*args, **kwargs)
        self.tensors = tensors

    def persistent_id(self, obj):
        if isinstance(obj, torch.Tensor):
            self.tensors.append(obj)
            return ''
        if isinstance(obj, LockType):
            return ''
        if isinstance(obj, CFuture) or is_rref_instance(obj):
            return ''
        if isinstance(obj, CAwait):
            return ''
        if isinstance(obj, torch.cuda.Event):
            return ''
        if isinstance(obj, threading.Thread):
            return ''
        return None