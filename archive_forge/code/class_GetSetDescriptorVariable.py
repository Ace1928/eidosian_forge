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
class GetSetDescriptorVariable(VariableTracker):

    def __init__(self, desc, **kwargs):
        super().__init__(**kwargs)
        self.desc = desc

    def var_getattr(self, tx, name):
        if name == '__get__' and self.source:
            from .builder import VariableBuilder
            return VariableBuilder(tx, AttrSource(self.source, '__get__'))(self.desc.__get__)
        else:
            return super().var_getattr(tx, name)

    def is_python_constant(self):
        return True

    def as_python_constant(self):
        return self.desc