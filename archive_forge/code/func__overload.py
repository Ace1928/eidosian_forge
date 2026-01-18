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
def _overload(func):
    _check_overload_body(func)
    qual_name = _qualified_name(func)
    global _overloaded_fns
    fn_overload_list = _overloaded_fns.get(qual_name)
    if fn_overload_list is None:
        fn_overload_list = []
        _overloaded_fns[qual_name] = fn_overload_list
    fn_overload_list.append(func)
    return func