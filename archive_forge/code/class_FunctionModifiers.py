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
class FunctionModifiers:
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """
    UNUSED = 'unused (ignored and replaced with raising of an exception)'
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = 'export (compile this function even if nothing calls it)'
    DEFAULT = 'default (compile if called from a exported function / forward)'
    COPY_TO_SCRIPT_WRAPPER = 'if this method is not scripted, copy the python method onto the scripted model'
    _DROP = '_drop (function is fully ignored, declaration can be unscriptable)'