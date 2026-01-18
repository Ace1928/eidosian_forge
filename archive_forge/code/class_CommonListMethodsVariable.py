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
class CommonListMethodsVariable(BaseListVariable):
    """
    Implement methods common to List and other List-like things
    """

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        if name == 'append' and self.mutable_local:
            assert not kwargs
            arg, = args
            tx.replace_all(self, self.modified(self.items + [arg]))
            return ConstantVariable.create(None)
        elif name == 'extend' and self.mutable_local and args and args[0].has_unpack_var_sequence(tx):
            assert not kwargs
            arg, = args
            return tx.replace_all(self, self.modified(list(self.items) + list(arg.unpack_var_sequence(tx))))
        elif name == 'insert' and self.mutable_local:
            assert not kwargs
            idx, value = args
            items = list(self.items)
            items.insert(idx.as_python_constant(), value)
            return tx.replace_all(self, self.modified(items))
        elif name == 'pop' and self.mutable_local:
            assert not kwargs
            items = list(self.items)
            result = items.pop(*[a.as_python_constant() for a in args])
            tx.replace_all(self, self.modified(items))
            return result
        elif name == 'clear' and self.mutable_local:
            assert not kwargs and (not args)
            return tx.replace_all(self, self.modified([]))
        elif name == '__setitem__' and self.mutable_local and args and args[0].is_python_constant():
            assert not kwargs
            key, value = args
            items = list(self.items)
            if isinstance(key, SliceVariable):
                items[key.as_python_constant()] = list(value.items)
            else:
                items[key.as_python_constant()] = value
            result = self.modified(items)
            return tx.replace_all(self, result)
        elif name == 'copy':
            assert not kwargs
            assert not args
            items = list(self.items)
            return self.modified(items, mutable_local=MutableLocal())
        else:
            return super().call_method(tx, name, args, kwargs)