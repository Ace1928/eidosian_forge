import functools
import inspect
import itertools
import types
from typing import Dict, List
import torch
from .. import variables
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented, Unsupported
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import make_cell
from .base import typestr, VariableTracker
class WrappedUserFunctionVariable(UserFunctionVariable):

    def __init__(self, wrapped, context, **kwargs):
        kwargs.pop('fn', None)
        kwargs.pop('obj', None)
        super().__init__(wrapped.fn, **kwargs)
        self.wrapped = wrapped
        self.context = context

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        self.context.enter(tx)
        result = super().call_function(tx, args, kwargs)
        self.context.exit(tx)
        return result