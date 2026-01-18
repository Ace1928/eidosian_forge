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
class InspectSignatureVariable(VariableTracker):
    """represents inspect.signature(...)"""

    @staticmethod
    def create(callable, **kwargs):
        if kwargs:
            unimplemented(f'inspect.signature with {kwargs}')
        return InspectSignatureVariable(callable)

    def __init__(self, inspected: VariableTracker, **kwargs):
        super().__init__(**kwargs)
        self.inspected = inspected

    def var_getattr(self, tx, name: str) -> 'VariableTracker':
        if name == 'parameters':
            return variables.ConstDictVariable({name: InspectParameterVariable() for name in self.inspected.inspect_parameter_names()}, user_cls=dict)
        return super().var_getattr(tx, name)