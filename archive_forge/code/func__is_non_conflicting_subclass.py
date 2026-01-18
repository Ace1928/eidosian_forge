import collections
import functools
import inspect
import operator
import types
from typing import Dict, List, Optional
import torch
import torch.fx
from ..._guards import Source
from .. import polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..source import AttrSource, GetItemSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .functions import UserFunctionVariable, UserMethodVariable
@classmethod
def _is_non_conflicting_subclass(cls, user_cls: type, python_cls: type):
    """Ensures user_cls inherits from python_cls (e.g. list) and does not override any methods on python_cls"""
    if not istype(user_cls, type) or user_cls.__bases__ != (python_cls,) or user_cls.__mro__ != (user_cls, python_cls, object):
        return False
    return not any((hasattr(python_cls, name) or name in cls._disallowed_names for name in set(user_cls.__dict__.keys()) - cls._allowed_names))