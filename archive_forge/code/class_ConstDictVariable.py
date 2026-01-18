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
class ConstDictVariable(VariableTracker):

    def __init__(self, items, user_cls, **kwargs):
        super().__init__(**kwargs)
        assert not any((isinstance(x, VariableTracker) for x in items))
        self.items = items
        self.user_cls = user_cls

    def as_proxy(self):
        return {k: v.as_proxy() for k, v in self.items.items()}

    def as_python_constant(self):
        return {k: v.as_python_constant() for k, v in self.items.items()}

    def python_type(self):
        return self.user_cls

    def reconstruct(self, codegen):
        if self.user_cls is collections.OrderedDict:
            codegen.extend_output([codegen.create_load_python_module(collections, True), codegen.create_load_attr('OrderedDict')])
        for key in self.items.keys():
            if istensor(key):
                codegen.append_output(codegen.create_load_global(global_key_name(key), True, add=True))
                codegen.extend_output(create_call_function(0, False))
            else:
                codegen.append_output(codegen.create_load_const(key))
            codegen(self.items[key])
        if self.user_cls is collections.OrderedDict:
            return [create_instruction('BUILD_MAP', arg=len(self.items)), *create_call_function(1, False)]
        else:
            return [create_instruction('BUILD_MAP', arg=len(self.items))]

    def getitem_const(self, arg: VariableTracker):
        return self.items[ConstDictVariable.get_key(arg)]

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from . import ConstantVariable, ListIteratorVariable, ListVariable, TupleVariable
        val = self.items
        if name == '__getitem__':
            assert len(args) == 1
            return self.getitem_const(args[0])
        elif name == 'items':
            assert not (args or kwargs)
            return TupleVariable([TupleVariable(items=[ConstDictVariable._key_to_var(tx, k), v]) for k, v in val.items()])
        elif name == 'keys':
            assert not (args or kwargs)
            return SetVariable(items=[ConstDictVariable._key_to_var(tx, k) for k in val.keys()], mutable_local=MutableLocal())
        elif name == 'values':
            assert not (args or kwargs)
            return TupleVariable(list(val.values()))
        elif name == 'copy':
            assert not (args or kwargs)
            return self.modifed(self.items.copy(), mutable_local=MutableLocal())
        elif name == '__len__':
            assert not (args or kwargs)
            return ConstantVariable.create(len(self.items))
        elif name == '__setitem__' and args and ConstDictVariable.is_valid_key(args[0]) and self.mutable_local:
            assert not kwargs and len(args) == 2
            k = ConstDictVariable.get_key(args[0])
            if istensor(k):
                tx.store_global_weakref(global_key_name(k), k)
            newval = dict(val)
            newval[k] = args[1]
            return tx.replace_all(self, self.modifed(newval))
        elif name in ('pop', 'get') and len(args) == 2 and (not kwargs) and ConstDictVariable.is_valid_key(args[0]) and (ConstDictVariable.get_key(args[0]) not in self.items):
            return args[1]
        elif name == 'get' and len(args) == 1 and (not kwargs) and ConstDictVariable.is_valid_key(args[0]) and (ConstDictVariable.get_key(args[0]) not in self.items):
            return ConstantVariable(None)
        elif name == 'pop' and args and ConstDictVariable.is_valid_key(args[0]) and self.mutable_local:
            newval = dict(val)
            result = newval.pop(ConstDictVariable.get_key(args[0]))
            tx.replace_all(self, self.modifed(newval))
            return result
        elif name == 'update' and len(args) == 1 and isinstance(args[0], ConstDictVariable) and self.mutable_local:
            newval = dict(val)
            newval.update(args[0].items)
            newval.update(kwargs)
            result = self.modifed(newval)
            return tx.replace_all(self, result)
        elif name == 'update' and len(args) == 1 and isinstance(args[0], (ListVariable, TupleVariable, ListIteratorVariable)) and self.mutable_local:
            newval = dict(val)
            for x in args[0].unpack_var_sequence(tx):
                k, v = x.unpack_var_sequence(tx)
                assert ConstDictVariable.is_valid_key(k)
                newval[ConstDictVariable.get_key(k)] = v
            newval.update(kwargs)
            result = self.modifed(newval)
            return tx.replace_all(self, result)
        elif name in ('get', '__getattr__') and args and ConstDictVariable.is_valid_key(args[0]) and (ConstDictVariable.get_key(args[0]) in self.items):
            return self.items[ConstDictVariable.get_key(args[0])]
        elif name == '__contains__' and args and ConstDictVariable.is_valid_key(args[0]):
            return ConstantVariable.create(ConstDictVariable.get_key(args[0]) in self.items)
        else:
            return super().call_method(tx, name, args, kwargs)

    def modifed(self, items, **options):
        """a copy of self with different items"""
        return self.clone(items=items, **options)

    def unpack_var_sequence(self, tx):
        val = self.items
        result = [ConstDictVariable._key_to_var(tx, k) for k in val.keys()]
        return result

    @classmethod
    def get_key(cls, arg: VariableTracker):
        if isinstance(arg, TensorVariable) and arg.specialized_value is not None:
            return arg.specialized_value
        else:
            return arg.as_python_constant()

    @classmethod
    def is_valid_key(cls, key):
        return key.is_python_constant() or (isinstance(key, TensorVariable) and key.specialized_value is not None) or (isinstance(key, ConstantVariable) and key.python_type() is torch.dtype)

    @classmethod
    def _key_to_var(cls, tx, key, **options):
        from .builder import VariableBuilder
        if istensor(key):
            return VariableBuilder(tx, GlobalWeakRefSource(global_key_name(key)))(key)
        else:
            assert ConstantVariable.is_literal(key)
            return ConstantVariable.create(key, **options)