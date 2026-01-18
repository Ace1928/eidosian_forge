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
def _autowrap_check(patcher: _Patcher, frame_dict: Dict[str, Any], function_ids: Set[int]):
    """
    Some methods, like `math.sqrt` are common enough we want to automatically wrap them as we see them.
    This method searches a scope for them and patches them if found.
    """
    if patcher.visit_once(frame_dict):
        for name, value in frame_dict.items():
            if not name.startswith('_') and callable(value) and (id(value) in function_ids):
                patcher.patch(frame_dict, name, _create_wrapped_func(value))