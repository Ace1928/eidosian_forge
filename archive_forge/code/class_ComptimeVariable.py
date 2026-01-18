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
class ComptimeVariable(VariableTracker):
    """
    This variable is special, it lets you execute arbitrary code at
    Dynamo compile time
    """

    def reconstruct(self, codegen):
        raise NotImplementedError('comptime is special form')

    def var_getattr(self, tx, name: str) -> 'VariableTracker':
        from ..comptime import comptime
        from .functions import UserFunctionVariable
        return UserFunctionVariable(getattr(comptime, name), source=AttrSource(self.source, name))

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from ..comptime import ComptimeContext
        assert not kwargs
        assert len(args) == 1
        fn = args[0]
        if isinstance(fn, UserFunctionVariable):
            fn.get_function()(ComptimeContext(tx))
        elif isinstance(fn, NestedUserFunctionVariable):
            code = fn.get_code()
            assert not fn.closure, f'comptime function must not have free variables, but these variables were free: {code.co_freevars}'
            func = types.FunctionType(code, fn.f_globals, fn.fn_name.as_python_constant(), tuple(fn.defaults.items) if fn.defaults else None, tuple())
            func(ComptimeContext(tx))
        else:
            raise RuntimeError(f'unsupported argument to comptime: {type(fn)}')
        return variables.ConstantVariable.create(None)