import collections
import copy
import enum
import functools
import inspect
import pickle
import warnings
from typing import Any, Callable, Dict, List, Set, Tuple, Union
import torch
import torch._jit_internal as _jit_internal
from torch._classes import classes
from torch._jit_internal import _qualified_name
from torch.jit._builtins import _register_builtin
from torch.jit._fuser import _graph_for, _script_method_graph_for
from torch.jit._monkeytype_config import (
from torch.jit._recursive import (
from torch.jit._state import (
from torch.jit.frontend import get_default_args, get_jit_class_def, get_jit_def
from torch.nn import Module
from torch.overrides import (
from torch.package import PackageExporter, PackageImporter
from torch.utils import set_module
from ._serialization import validate_map_location
def _get_overloads(obj):
    existing_compiled_fns = _try_get_jit_cached_overloads(obj)
    qual_name = _qualified_name(obj)
    uncompiled_overloads = _jit_internal._get_fn_overloads(qual_name)
    if uncompiled_overloads is None:
        return existing_compiled_fns
    if obj in uncompiled_overloads:
        raise RuntimeError(_jit_internal.get_overload_no_implementation_error_message('function', obj))
    compiled_fns = []
    for overload_fn in uncompiled_overloads:
        compiled_fns.append(_compile_function_with_overload(overload_fn, qual_name, obj))
    if existing_compiled_fns:
        compiled_fns = existing_compiled_fns + compiled_fns
    _set_jit_overload_cache(obj, compiled_fns)
    _jit_internal._clear_fn_overloads(qual_name)
    return compiled_fns