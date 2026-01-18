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
class DataClassVariable(ConstDictVariable):
    """
    This is a bit of a hack to deal with
    transformers.file_utils.ModelOutput() from huggingface.

    ModelOutput causes trouble because it a a mix of a dataclass and a
    OrderedDict and it calls super() methods implemented in C.
    """
    include_none = False

    @staticmethod
    @functools.lru_cache(None)
    def _patch_once():
        try:
            from transformers.file_utils import ModelOutput
            for obj in ModelOutput.__dict__.values():
                if callable(obj):
                    skip_code(obj.__code__)
        except ImportError:
            pass
        try:
            from diffusers.utils import BaseOutput
            for obj in BaseOutput.__dict__.values():
                if callable(obj):
                    skip_code(obj.__code__)
        except ImportError:
            pass

    @staticmethod
    def is_matching_cls(cls):
        return _is_matching_transformers_cls(cls) or _is_matching_diffusers_cls(cls)

    @classmethod
    def is_matching_object(cls, obj):
        return cls.is_matching_cls(type(obj))

    @classmethod
    def create(cls, user_cls, args, kwargs, options):
        DataClassVariable._patch_once()
        skip_code(user_cls.__init__.__code__)
        keys = [f.name for f in dataclasses.fields(user_cls)]
        bound = inspect.signature(user_cls).bind(*args, **kwargs)
        bound.apply_defaults()
        assert set(bound.arguments.keys()) == set(keys)
        items = {}
        for key in keys:
            val = bound.arguments[key]
            if isinstance(val, VariableTracker):
                items[key] = val
            elif cls.include_none:
                assert variables.ConstantVariable.is_literal(val)
                items[key] = variables.ConstantVariable.create(val)
            else:
                assert val is None, f'unexpected {val}'
        if len(items) == 1 and (not isinstance(items[keys[0]], variables.TensorVariable)):
            unimplemented('DataClassVariable iterator constructor')
        return cls(items, user_cls, **options)

    @classmethod
    def wrap(cls, builder, obj):
        user_cls = type(obj)
        keys = [f.name for f in dataclasses.fields(user_cls)]
        excluded = []
        items = {}
        for key in keys:
            if hasattr(obj, key):
                val = getattr(obj, key)
                var = builder.__class__(tx=builder.tx, source=AttrSource(builder.source, key))(val)
                if val is not None or cls.include_none:
                    items[key] = var
                else:
                    excluded.append(var)
        return cls(items, user_cls)

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
        if name == '__getitem__':
            assert not kwargs and len(args) == 1
            index = args[0].as_python_constant()
            if isinstance(index, str):
                return self.items[index]
            else:
                return self.call_method(tx, 'to_tuple', [], {}).call_method(tx, '__getitem__', args, kwargs)
        elif name == 'to_tuple':
            assert not (args or kwargs)
            return variables.TupleVariable(list(self.items.values()))
        elif name == '__setattr__':
            name = '__setitem__'
        return super().call_method(tx, name, args, kwargs)

    def var_getattr(self, tx, name: str) -> 'VariableTracker':
        if name in self.items:
            return self.call_method(tx, '__getitem__', [variables.ConstantVariable.create(name)], {})
        elif not self.include_none:
            defaults = {f.name: f.default for f in dataclasses.fields(self.user_cls)}
            if name in defaults:
                assert variables.ConstantVariable.is_literal(defaults[name])
                return variables.ConstantVariable.create(defaults[name])
        super().var_getattr(tx, name)
    call_hasattr = _call_hasattr_customobj