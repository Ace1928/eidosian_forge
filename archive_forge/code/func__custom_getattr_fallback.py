import functools
import inspect
import itertools
import types
from contextlib import contextmanager, nullcontext
from typing import Dict, List
import torch.nn
from .. import skipfiles, variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented, UnspecializeRestartAnalysis, Unsupported
from ..guards import GuardBuilder, install_guard
from ..mutation_guard import GenerationTracker
from ..source import (
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .functions import invoke_and_store_as_constant
from .lists import SliceVariable
from .user_defined import UserDefinedObjectVariable
def _custom_getattr_fallback(self, base, tx, name, options):
    """Check for a __getattr__ and handle it specially if it is implemented"""
    if object_has_getattribute(base):
        unimplemented('torch.nn.Module with a custom __getattribute__ defined')
    getattr_fn = get_custom_getattr(base)
    if getattr_fn is None:
        return None
    if not isinstance(getattr_fn, types.FunctionType):
        unimplemented('torch.nn.Module with a non-function custom __getattr__')
    return variables.UserMethodVariable(getattr_fn, self, **options).call_function(tx, [variables.ConstantVariable.create(name)], {})