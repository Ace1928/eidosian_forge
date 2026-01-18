from inspect import signature, Signature
from typing import (
import ast
import builtins
import collections
import operator
import sys
from functools import cached_property
from dataclasses import dataclass, field
from types import MethodDescriptorType, ModuleType
from IPython.utils.docs import GENERATING_DOCUMENTATION
from IPython.utils.decorators import undoc
def _unbind_method(func: Callable) -> Union[Callable, None]:
    """Get unbound method for given bound method.

    Returns None if cannot get unbound method, or method is already unbound.
    """
    owner = getattr(func, '__self__', None)
    owner_class = type(owner)
    name = getattr(func, '__name__', None)
    instance_dict_overrides = getattr(owner, '__dict__', None)
    if owner is not None and name and (not instance_dict_overrides or (instance_dict_overrides and name not in instance_dict_overrides)):
        return getattr(owner_class, name)
    return None