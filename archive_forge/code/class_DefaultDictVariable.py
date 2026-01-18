import collections
import dataclasses
import functools
import inspect
import sys
from typing import Any, Dict, List, Optional
import torch
import torch.fx
from .. import variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..eval_frame import skip_code
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard, make_dupe_guard
from ..source import AttrSource, GetItemSource, GlobalWeakRefSource
from ..utils import global_key_name, istensor, istype, iter_contains
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .tensor import TensorVariable
class DefaultDictVariable(ConstDictVariable):

    def __init__(self, items, user_cls, default_factory=None, **kwargs):
        super().__init__(items, user_cls, **kwargs)
        assert user_cls is collections.defaultdict
        self.default_factory = default_factory

    def is_python_constant(self):
        if self.default_factory not in [list, tuple, dict] and (not self.items):
            return False
        return super().is_python_constant()

    @staticmethod
    def is_supported_arg(arg):
        if isinstance(arg, variables.BuiltinVariable):
            return arg.fn in [list, tuple, dict]
        else:
            return isinstance(arg, variables.functions.BaseUserFunctionVariable)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if name == '__getitem__':
            k = ConstDictVariable.get_key(args[0])
            if k in self.items:
                return self.getitem_const(args[0])
            elif self.default_factory is None:
                raise KeyError(f'{k}')
            else:
                if istensor(k):
                    tx.store_global_weakref(global_key_name(k), k)
                new_val = dict(self.items)
                default_var = self.default_factory.call_function(tx, [], {})
                new_val[k] = default_var
                tx.replace_all(self, self.modifed(new_val))
                return default_var
        else:
            return super().call_method(tx, name, args, kwargs)