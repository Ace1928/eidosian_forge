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
class RestrictedListSubclassVariable(ListVariable):
    """
    This is a special case of UserDefinedObjectVariable where:
        1) The user subclasses list
        2) None of the list methods are overriden, merely some new methods are added

    In these cases, we can prevent graph breaks by not using the general
    UserDefinedObjectVariable machinery and instead treating it like
    a ListVariable.
    """
    _nonvar_fields = {'user_cls', 'user_cls_source', *ListVariable._nonvar_fields}
    _allowed_names = {'__call__', '__module__', '__dict__', '__doc__', '__name__', '__qualname__'}
    _disallowed_names = {'__getattribute__', '__getattr__', '__setattr__'}

    @classmethod
    def _is_non_conflicting_subclass(cls, user_cls: type, python_cls: type):
        """Ensures user_cls inherits from python_cls (e.g. list) and does not override any methods on python_cls"""
        if not istype(user_cls, type) or user_cls.__bases__ != (python_cls,) or user_cls.__mro__ != (user_cls, python_cls, object):
            return False
        return not any((hasattr(python_cls, name) or name in cls._disallowed_names for name in set(user_cls.__dict__.keys()) - cls._allowed_names))

    @classmethod
    def is_matching_cls(cls, user_cls: type):
        return cls._is_non_conflicting_subclass(user_cls, list)

    def __init__(self, items, *, user_cls: type, user_cls_source: Source, **kwargs):
        super().__init__(items=items, **kwargs)
        self.user_cls = user_cls
        self.user_cls_source = user_cls_source
        assert istype(user_cls, type)
        assert isinstance(user_cls_source, Source)

    def python_type(self):
        return self.user_cls

    def as_proxy(self):
        return [x.as_proxy() for x in self.items]

    def as_python_constant(self):
        raise NotImplementedError()

    def is_python_constant(self):
        return False

    @property
    def value(self):
        raise AttributeError('value')

    def modified(self, items, **kwargs):
        return type(self)(items, user_cls=self.user_cls, user_cls_source=self.user_cls_source, **kwargs)

    def reconstruct(self, codegen):
        codegen(self.user_cls_source)
        return super().reconstruct(codegen) + create_call_function(1, True)

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        if name in self.user_cls.__dict__:
            method = self.user_cls.__dict__[name]
            if isinstance(method, types.FunctionType):
                source = AttrSource(self.user_cls_source, name)
                return UserMethodVariable(method, self, source=source).call_function(tx, args, kwargs)
            unimplemented(f'RestrictedListSubclassVariable method {self.user_cls.__name__}.{name}')
        return super().call_method(tx, name, args, kwargs)

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        return self.call_method(tx, '__call__', args, kwargs)