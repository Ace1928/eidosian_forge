import builtins
import collections
import copy
import dataclasses
import functools
import inspect
import itertools
import math
import operator
import sys
import types
import warnings
from collections import defaultdict
from typing import Any, Callable, cast, Dict, List, Optional, Set, Union
import torch
import torch._functorch.deprecated as deprecated_func
from torch.fx._symbolic_trace import is_fx_tracing
from . import config
from .external_utils import is_compiling
from .utils import hashable, is_safe_constant, NP_SUPPORTED_MODULES
def _find_torch_objects(module):
    if any((module.__name__.startswith(mod_name) for mod_name in config.allowed_functions_module_string_ignorelist)):
        return
    torch_object_ids[id(module)] = module.__name__
    for name, obj in list(module.__dict__.items()):
        if id(obj) not in torch_object_ids:
            import torch._ops
            if isinstance(obj, torch._ops.HigherOrderOperator):
                continue
            if obj in (torch.func.grad, deprecated_func.grad, torch.func.vmap, deprecated_func.vmap, torch.nn.functional.triplet_margin_with_distance_loss, torch.cond):
                continue
            if isinstance(obj, types.ModuleType):
                if obj.__name__.startswith('torch.') and _is_allowed_module_prefix(obj):
                    torch_object_ids[id(obj)] = f'{module.__name__}.{name}'
                    _find_torch_objects(obj)
            elif _is_allowed_module_prefix(obj):
                if record:
                    heuristic_record_if_ctx_manager(obj, module, name)
                    heuristic_record_if_in_graph_function(obj, module, name)
                torch_object_ids[id(obj)] = f'{module.__name__}.{name}'
            elif inspect.getmodule(obj) is None and (not is_safe_constant(obj)):
                if record:
                    heuristic_record_if_ctx_manager(obj, module, name)
                    heuristic_record_if_in_graph_function(obj, module, name)
                torch_object_ids[id(obj)] = f'{module.__name__}.{name}'