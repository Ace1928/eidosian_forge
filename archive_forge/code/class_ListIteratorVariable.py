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
class ListIteratorVariable(VariableTracker):

    def __init__(self, items, index: int=0, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(items, list)
        self.items = items
        self.index = index

    def __repr__(self):
        return f'{self.__class__.__name__}({repr(self.items)}, index={repr(self.index)})'

    def next_variables(self, tx):
        assert self.mutable_local
        if self.index >= len(self.items):
            raise StopIteration()
        next_iter = ListIteratorVariable(self.items, self.index + 1, mutable_local=MutableLocal())
        tx.replace_all(self, next_iter)
        return (self.items[self.index], next_iter)

    def as_python_constant(self):
        if self.index > 0:
            raise NotImplementedError()
        return iter([x.as_python_constant() for x in self.items])

    def unpack_var_sequence(self, tx):
        return list(self.items[self.index:])

    def reconstruct(self, codegen):
        remaining_items = self.items[self.index:]
        codegen.foreach(remaining_items)
        return [create_instruction('BUILD_TUPLE', arg=len(remaining_items)), create_instruction('GET_ITER')]