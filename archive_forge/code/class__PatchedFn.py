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
class _PatchedFn(NamedTuple):
    frame_dict: Any
    fn_name: str
    orig_fn: Any

    def revert(self):
        raise NotImplementedError()