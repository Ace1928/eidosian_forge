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
def _check_overload_body(func):
    try:
        parsed_def = parse_def(func)
    except OSError as e:
        warnings.warn(f'Unable to retrieve source for @torch.jit._overload function: {func}.')
        return
    body = parsed_def.ast.body[0].body

    def is_pass(x):
        return isinstance(x, ast.Pass)

    def is_ellipsis(x):
        return isinstance(x, ast.Expr) and isinstance(x.value, ast.Ellipsis)
    if len(body) != 1 or not (is_pass(body[0]) or is_ellipsis(body[0])):
        msg = 'Only `pass` statement or `...` can be the body of overload declaration:\n'
        msg += '\n'.join(parsed_def.source.split('\n')[:3])
        msg += ' <- Expecting `pass` or `...` here!\n' + _OVERLOAD_EXAMPLE
        raise RuntimeError(msg)