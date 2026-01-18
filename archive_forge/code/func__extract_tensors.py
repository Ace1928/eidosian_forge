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
def _extract_tensors(obj):
    """
    This function is exclusively called from C++.
    See ``torch/csrc/jit/python/python_ivalue.h``.

    It extracts the tensors contained in the given object, through pickling.
    """
    tensors: List[torch.Tensor] = []
    extractor = _TensorExtractor(io.BytesIO(), protocol=-1, tensors=tensors)
    extractor.dump(obj)
    return tensors