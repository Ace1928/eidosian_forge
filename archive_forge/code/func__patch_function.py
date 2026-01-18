import builtins
import copy
import functools
import inspect
import math
import os
import warnings
import collections
from itertools import chain
from types import CodeType, FunctionType, ModuleType
from typing import (
import torch
import torch.utils._pytree as pytree
from torch._C import ScriptObject  # type: ignore[attr-defined]
from ._compatibility import compatibility
from .graph import _PyTreeCodeGen, _PyTreeInfo, Graph
from .graph_module import GraphModule
from .node import Argument, base_types, map_aggregate
from .proxy import ParameterProxy, Proxy, TracerBase, Scope, ScopeContextManager
def _patch_function(fn: FunctionType, nargs: int) -> FunctionType:
    co = fn.__code__
    co_flags = co.co_flags & ~HAS_VARSTUFF
    co_args: tuple
    if hasattr(co, 'co_qualname'):
        co_args = (nargs, 0, 0, co.co_nlocals, co.co_stacksize, co_flags, co.co_code, co.co_consts, co.co_names, co.co_varnames, co.co_filename, co.co_name, co.co_qualname, co.co_firstlineno, co.co_lnotab, co.co_exceptiontable, co.co_freevars, co.co_cellvars)
    elif hasattr(co, 'co_posonlyargcount'):
        co_args = (nargs, 0, 0, co.co_nlocals, co.co_stacksize, co_flags, co.co_code, co.co_consts, co.co_names, co.co_varnames, co.co_filename, co.co_name, co.co_firstlineno, co.co_lnotab, co.co_freevars, co.co_cellvars)
    else:
        co_args = (nargs, 0, co.co_nlocals, co.co_stacksize, co_flags, co.co_code, co.co_consts, co.co_names, co.co_varnames, co.co_filename, co.co_name, co.co_firstlineno, co.co_lnotab, co.co_freevars, co.co_cellvars)
    new_code = CodeType(*co_args)
    return FunctionType(new_code, fn.__globals__, fn.__name__, fn.__defaults__, fn.__closure__)