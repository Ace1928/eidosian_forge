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
def _compile_function_with_overload(overload_fn, qual_name, impl_fn):
    overload_decl = get_jit_def(overload_fn, overload_fn.__name__).decl()
    overload_signature = torch.jit.annotations.get_signature(overload_fn, None, None, inspect.ismethod(overload_fn))
    impl_ast = get_jit_def(impl_fn, impl_fn.__name__)
    overload_defaults = get_default_args(overload_fn)
    implementation_defaults = get_default_args(impl_fn)
    _rcb = _jit_internal.createResolutionCallbackFromClosure(impl_fn)
    _check_overload_defaults(implementation_defaults, overload_defaults, overload_decl.range())
    fn = torch._C._jit_script_compile_overload(qual_name, overload_decl, impl_ast, _rcb, implementation_defaults, overload_signature)
    return fn