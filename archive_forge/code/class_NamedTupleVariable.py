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
class NamedTupleVariable(TupleVariable):

    def __init__(self, items, tuple_cls, **kwargs):
        super().__init__(items, **kwargs)
        self.tuple_cls = tuple_cls

    def python_type(self):
        return self.tuple_cls

    def as_python_constant(self):
        return self.python_type()(*[x.as_python_constant() for x in self.items])

    def reconstruct(self, codegen):
        create_fn = getattr(self.tuple_cls, '_make', self.tuple_cls)
        codegen.append_output(codegen._create_load_const(create_fn))
        codegen.foreach(self.items)
        return [create_instruction('BUILD_TUPLE', arg=len(self.items))] + create_call_function(1, True)

    def var_getattr(self, tx, name):

        def check_and_create_method():
            method = inspect.getattr_static(self.tuple_cls, name, None)
            if isinstance(method, classmethod):
                return UserMethodVariable(method.__func__, variables.UserDefinedClassVariable(self.tuple_cls))
            elif isinstance(method, staticmethod):
                return UserFunctionVariable(method.__func__)
            elif inspect.isfunction(method):
                return UserMethodVariable(method, self)
            else:
                return None
        fields = namedtuple_fields(self.tuple_cls)
        if name not in fields:
            method = check_and_create_method()
            if not method:
                super().var_getattr(tx, name)
            return method
        return self.items[fields.index(name)]

    def call_hasattr(self, tx, name: str) -> 'VariableTracker':
        fields = namedtuple_fields(self.tuple_cls)
        return variables.ConstantVariable.create(name in fields)