import collections
import dataclasses
import functools
import inspect
import itertools
import operator
import sys
import types
from typing import Dict, List
import torch._C
import torch._numpy as tnp
from .. import config, polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, TypeSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .dicts import DefaultDictVariable
from .functions import (
from .user_defined import UserDefinedObjectVariable
class MethodWrapperVariable(VariableTracker):

    def __init__(self, method_wrapper, **kwargs):
        super().__init__(**kwargs)
        self.method_wrapper = method_wrapper

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if is_tensor_base_attr_getter(self.method_wrapper) and isinstance(args[0], variables.TensorVariable):
            assert len(args) == 1 and len(kwargs) == 0
            return args[0].var_getattr(tx, self.method_wrapper.__self__.__name__)
        super().call_function(tx, args, kwargs)

    def is_python_constant(self):
        return True

    def as_python_constant(self):
        return self.method_wrapper