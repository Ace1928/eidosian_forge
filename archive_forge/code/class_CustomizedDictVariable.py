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
class CustomizedDictVariable(ConstDictVariable):

    @staticmethod
    def is_matching_cls(cls):
        if issubclass(cls, collections.OrderedDict) and cls.__init__ is collections.OrderedDict.__init__ and (not hasattr(cls, '__post_init__')):
            return True
        return _is_matching_transformers_cls(cls) or _is_matching_diffusers_cls(cls)

    @classmethod
    def is_matching_object(cls, obj):
        return cls.is_matching_cls(type(obj))

    @classmethod
    def create(cls, user_cls, args, kwargs, options):
        for attr_name in ('__init__', '__post_init__', '__setattr__', '__setitem__'):
            if hasattr(user_cls, attr_name):
                fn = getattr(user_cls, attr_name)
                assert callable(fn), f'expect callable attr {attr_name}'
                if hasattr(fn, '__code__'):
                    skip_code(fn.__code__)
        if not args and (not kwargs):
            raw_items = {}
        elif dataclasses.is_dataclass(user_cls):
            bound = inspect.signature(user_cls).bind(*args, **kwargs)
            bound.apply_defaults()
            raw_items = bound.arguments
        elif not args:
            raw_items = dict(kwargs)
        elif len(args) == 1 and isinstance(args[0], ConstDictVariable) and (not kwargs):
            raw_items = args[0].items
        else:
            unimplemented('custom dict init with args/kwargs unimplemented')
        items = {}
        for key in raw_items.keys():
            val = raw_items[key]
            if isinstance(val, VariableTracker):
                items[key] = val
            elif variables.ConstantVariable.is_literal(val):
                items[key] = variables.ConstantVariable.create(val)
            else:
                unimplemented('expect VariableTracker or ConstantVariable.is_literal')
        return cls(items, user_cls, **options)

    @classmethod
    def wrap(cls, builder, obj):
        raise NotImplementedError()

    def __init__(self, items, user_cls, **options):
        super().__init__(items, user_cls, **options)
        assert self.is_matching_cls(user_cls)

    def as_proxy(self):
        raise NotImplementedError()

    def reconstruct(self, codegen):
        codegen.extend_output([codegen._create_load_const(self.user_cls)])
        keys = tuple(self.items.keys())
        for key in keys:
            codegen(self.items[key])
        return codegen.create_call_function_kw(len(keys), keys, True)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        fn = getattr(self.user_cls, name)
        source = None if self.source is None else AttrSource(self.source, name)
        if hasattr(fn, '__objclass__') and fn.__objclass__ in (dict, collections.OrderedDict):
            return super().call_method(tx, name, args, kwargs)
        elif name in ('__getitem__', 'to_tuple', '__setitem__', '__setattr__'):
            return tx.inline_user_function_return(variables.UserFunctionVariable(fn, source=source), [self] + list(args), kwargs)
        unimplemented('custom dict: call_method unimplemented name=%s', name)

    def var_getattr(self, tx, name: str) -> 'VariableTracker':
        if name in self.items:
            return self.call_method(tx, '__getitem__', [variables.ConstantVariable.create(name)], {})
        super().var_getattr(tx, name)
    call_hasattr = _call_hasattr_customobj