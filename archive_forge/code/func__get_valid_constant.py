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
def _get_valid_constant(attr, v, owner_type):
    if isinstance(v, _constant_types):
        return v
    elif isinstance(v, (tuple, list)):
        return tuple((_get_valid_constant(attr, x, owner_type) for x in v))
    constants = ', '.join((torch.typename(typ) for typ in _constant_types))
    raise TypeError(textwrap.dedent(f"\n        '{torch.typename(type(v))}' object in attribute '{owner_type}.{attr}' is not a valid constant.\n        Valid constants are:\n        1. a nn.ModuleList\n        2. a value of type {{{constants}}}\n        3. a list or tuple of (2)\n        "))