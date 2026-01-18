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
class UserMethodVariable(UserFunctionVariable):
    """Some unsupported user-defined method"""

    def __init__(self, fn, obj, **kwargs):
        super().__init__(fn=fn, **kwargs)
        self.obj = obj

    def __str__(self):
        return f'{self.__class__.__name__}({self.fn}, {self.obj})'

    def self_args(self):
        return [self.obj]

    def python_type(self):
        return types.MethodType

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if tx.output.is_root_tracer() and isinstance(self.obj, variables.NNModuleVariable):
            module_attr = getattr(self.fn, '__module__', '')
            if module_attr is not None and module_attr.startswith('torch.nn.') or self.is_constant:
                return self.obj.call_method(tx, self.fn.__name__, args, kwargs, constant=self.is_constant)
        return super().call_function(tx, args, kwargs)

    def inspect_parameter_names(self):
        return super().inspect_parameter_names()[1:]