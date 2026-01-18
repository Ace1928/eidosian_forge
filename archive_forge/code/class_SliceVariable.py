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
class SliceVariable(BaseListVariable):

    def __init__(self, items, **kwargs):
        items_to_map = items
        start, stop, step = [variables.ConstantVariable.create(None)] * 3
        if len(items_to_map) == 1:
            stop, = items_to_map
        elif len(items_to_map) == 2:
            start, stop = items_to_map
        elif len(items_to_map) == 3:
            start, stop, step = items_to_map
        else:
            raise AssertionError()
        if isinstance(start, variables.TensorVariable) or isinstance(stop, variables.TensorVariable):
            unimplemented('Dynamic slicing on data-dependent value is not supported')
        super().__init__([start, stop, step], **kwargs)

    def as_proxy(self):
        return slice(*self._as_proxy())

    def python_type(self):
        return slice

    def as_python_constant(self):
        return slice(*[guard_if_dyn(x) for x in self.items])

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction('BUILD_SLICE', arg=len(self.items))]

    def var_getattr(self, tx, name):
        fields = ['start', 'stop', 'step']
        if name not in fields:
            unimplemented(f'slice.{name}')
        return self.items[fields.index(name)]