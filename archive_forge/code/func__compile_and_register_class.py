import collections
import functools
import inspect
import sys
import textwrap
import types
import warnings
from typing import Dict, List, Set, Type
import torch
import torch._jit_internal as _jit_internal
from torch._sources import fake_range
from torch.jit._builtins import _find_builtin
from torch.jit._check import AttributeTypeIsSupportedChecker
from torch.jit._state import _add_script_class, _get_script_class, _python_cu
from torch.jit.frontend import (
from torch.nn import Module
def _compile_and_register_class(obj, rcb, qualified_name):
    script_class = _get_script_class(obj)
    if not script_class:
        ast = get_jit_class_def(obj, obj.__name__)
        defaults = torch.jit.frontend.get_default_args_for_class(obj)
        script_class = torch._C._jit_script_class_compile(qualified_name, ast, defaults, rcb)
        _add_script_class(obj, script_class)
    return script_class