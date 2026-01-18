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
def _create_named_tuple(t, unqual_name: str, field_names: List[str], defaults: Tuple[Any, ...]):
    TupleType = collections.namedtuple(unqual_name, field_names, defaults=defaults)
    return TupleType(*t)