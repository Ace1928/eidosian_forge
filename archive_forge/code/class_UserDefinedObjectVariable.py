import collections
import contextlib
import functools
import importlib
import inspect
import itertools
import random
import sys
import threading
import types
from typing import Dict, List
import torch._dynamo.config
import torch.nn
from torch._guards import TracingContext
from .. import variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, RandomValueSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .ctx_manager import GenericContextWrappingVariable, NullContextVariable
from .dicts import ConstDictVariable
class UserDefinedObjectVariable(UserDefinedVariable):
    """
    Mostly objects of defined type.  Catch-all for something where we only know the type.
    """
    _nonvar_fields = {'value', 'value_type', *UserDefinedVariable._nonvar_fields}

    def __init__(self, value, value_type=None, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.value_type = value_type or type(value)
        assert type(value) is self.value_type

    def __str__(self):
        inner = self.value_type.__name__
        if inner in ['builtin_function_or_method', 'getset_descriptor', 'method_descriptor', 'method']:
            inner = str(getattr(self.value, '__name__', None))
        return f'{self.__class__.__name__}({inner})'

    def python_type(self):
        return self.value_type

    @staticmethod
    @functools.lru_cache(None)
    def _supported_random_functions():
        fns = {random.random, random.randint, random.randrange, random.uniform}
        return fns

    def _maybe_get_baseclass_method(self, name):
        if name not in getattr(self.value, '__dict__', {}):
            try:
                return inspect.getattr_static(type(self.value), name)
            except AttributeError:
                pass
        return None

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from . import BuiltinVariable, ConstantVariable, TupleVariable, UserMethodVariable
        method = self._maybe_get_baseclass_method(name)
        if method is not None:
            if method is object.__init__:
                return ConstantVariable.create(None)
            if method is collections.OrderedDict.keys:
                assert not (args or kwargs)
                assert self.source
                keys = list(self.value.keys())
                assert all(map(ConstantVariable.is_literal, keys))
                install_guard(self.source.make_guard(GuardBuilder.ODICT_KEYS))
                return TupleVariable([ConstantVariable.create(k) for k in keys])
            if method in (collections.OrderedDict.__contains__, dict.__contains__) and len(args) == 1 and isinstance(args[0], (ConstantVariable, BuiltinVariable)) and (inspect.getattr_static(type(self.value), 'keys') in (collections.OrderedDict.keys, dict.keys)):
                assert not kwargs
                assert self.source
                install_guard(self.source.make_guard(GuardBuilder.ODICT_KEYS))
                return ConstantVariable.create(args[0].as_python_constant() in self.value)
            if method is collections.OrderedDict.items and isinstance(self.value, collections.OrderedDict):
                assert self.source
                assert not (args or kwargs)
                items = []
                keys = self.call_method(tx, 'keys', [], {})
                for key in keys.unpack_var_sequence(tx):
                    items.append(TupleVariable([key, self.odict_getitem(tx, key)]))
                return TupleVariable(items)
            if method is collections.OrderedDict.__getitem__ and len(args) == 1:
                assert not kwargs
                assert self.source
                return self.odict_getitem(tx, args[0])
            if isinstance(method, types.FunctionType):
                source = None if self.source is None else AttrSource(AttrSource(self.source, '__class__'), name)
                return UserMethodVariable(method, self, source=source).call_function(tx, args, kwargs)
            if method is list.__len__ and self.source and (not (args or kwargs)):
                install_guard(self.source.make_guard(GuardBuilder.LIST_LENGTH))
                return ConstantVariable(len(self.value))
        return super().call_method(tx, name, args, kwargs)

    def unpack_var_sequence(self, tx):
        if self.source and self._maybe_get_baseclass_method('__iter__') is list.__iter__ and (self._maybe_get_baseclass_method('__len__') is list.__len__) and (self._maybe_get_baseclass_method('__getitem__') is list.__getitem__):
            install_guard(self.source.make_guard(GuardBuilder.LIST_LENGTH))
            return [variables.LazyVariableTracker.create(self.value[k], source=GetItemSource(self.source, k)) for k in range(len(self.value))]
        return super().unpack_var_sequence(tx)

    def is_supported_random(self):
        try:
            return self.value in self._supported_random_functions()
        except TypeError:
            return False

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from .. import trace_rules
        from .builder import VariableBuilder
        if self.is_supported_random() and all((k.is_python_constant() for k in args)) and all((v.is_python_constant() for v in kwargs.values())):
            args = [x.as_python_constant() for x in args]
            kwargs = {k: v.as_python_constant() for k, v in kwargs.items()}
            random_call_index = len(tx.random_calls)
            example_value = self.value(*args, **kwargs)
            source = RandomValueSource(random_call_index)
            tx.random_calls.append((self.value, args, kwargs))
            return VariableBuilder(tx, source).wrap_unspecialized_primitive(example_value)
        elif istype(self.value, types.MethodType):
            func = self.value.__func__
            obj = self.value.__self__
            if func is torch.utils._contextlib._DecoratorContextManager.clone and trace_rules.lookup(obj.__class__) == variables.TorchCtxManagerClassVariable and (not (args or kwargs)):
                return variables.TorchCtxManagerClassVariable(obj.__class__).call_function(tx, args, kwargs)
            if func is torch.autograd.grad_mode.inference_mode.clone and obj.__class__ is torch.autograd.grad_mode.inference_mode:
                var = variables.ConstantVariable(obj.mode)
                return variables.TorchCtxManagerClassVariable(obj.__class__).call_function(tx, [var], kwargs)
        elif istype(self.value, functools.partial) and is_allowed(self.value.func) and all((variables.ConstantVariable.is_literal(v) for v in itertools.chain(self.value.args, self.value.keywords.values()))):
            if self.source:
                install_guard(AttrSource(self.source, 'func').make_guard(GuardBuilder.ID_MATCH), AttrSource(self.source, 'args').make_guard(GuardBuilder.CONSTANT_MATCH), AttrSource(self.source, 'keywords').make_guard(GuardBuilder.CONSTANT_MATCH))
            partial_args = [variables.ConstantVariable.create(v) for v in self.value.args]
            partial_args.extend(args)
            partial_kwargs = {k: variables.ConstantVariable.create(v) for k, v in self.value.keywords.items()}
            partial_kwargs.update(kwargs)
            if is_utils_checkpoint(self.value.func):
                return build_checkpoint_variable().call_function(tx, partial_args, partial_kwargs)
            return variables.TorchVariable(self.value.func).call_function(tx, partial_args, partial_kwargs)
        elif callable(self.value):
            if self.source:
                install_guard(self.source.make_guard(GuardBuilder.FUNCTION_MATCH))
            return self.call_method(tx, '__call__', args, kwargs)
        return super().call_function(tx, args, kwargs)

    def _check_for_getattribute(self):
        if object_has_getattribute(self.value):
            unimplemented('UserDefinedObjectVariable with custom __getattribute__')

    def _check_for_getattr(self):
        return get_custom_getattr(self.value)

    def _getattr_static(self, name):
        if isinstance(self.value, torch.nn.Module) or '__slots__' in self.value.__class__.__dict__ or type(self.value) == threading.local:
            subobj = getattr(self.value, name)
        else:
            subobj = inspect.getattr_static(self.value, name)
        return subobj

    def var_getattr(self, tx, name):
        from . import ConstantVariable
        from .builder import VariableBuilder
        value = self.value
        source = AttrSource(self.source, name) if self.source else None
        self._check_for_getattribute()
        getattr_fn = self._check_for_getattr()

        class NO_SUCH_SUBOBJ:
            pass
        try:
            subobj = self._getattr_static(name)
        except AttributeError:
            subobj = NO_SUCH_SUBOBJ
            if isinstance(getattr_fn, types.FunctionType):
                return variables.UserMethodVariable(getattr_fn, self, source=source).call_function(tx, [ConstantVariable.create(name)], {})
            elif getattr_fn is not None:
                unimplemented('UserDefined with non-function __getattr__')
        if isinstance(subobj, property):
            return variables.UserMethodVariable(subobj.fget, self, source=source).call_function(tx, [], {})
        elif isinstance(subobj, torch.distributions.utils.lazy_property):
            subobj_var = UserDefinedObjectVariable(subobj, source=source)
            return variables.UserMethodVariable(subobj.__get__.__func__, subobj_var, source=source).call_function(tx, [self], {})
        elif isinstance(subobj, staticmethod):
            return variables.UserFunctionVariable(subobj.__get__(self.value), source=source)
        elif isinstance(subobj, classmethod):
            return variables.UserMethodVariable(subobj.__func__, self, source=source)
        elif isinstance(subobj, types.FunctionType) or (isinstance(subobj, types.MethodType) and isinstance(self.value, torch.nn.Module)):
            dynamic_subobj = getattr(self.value, name)
            while dynamic_subobj is subobj and hasattr(subobj, '_torchdynamo_inline'):
                subobj = subobj._torchdynamo_inline
                dynamic_subobj = subobj
                source = AttrSource(source, '_torchdynamo_inline') if source else None
            if isinstance(subobj, types.MethodType):
                if dynamic_subobj.__self__ is not self.value:
                    unimplemented('__self__ mismatch for bound method')
                func = subobj.__func__
                source = AttrSource(source, '__func__') if source else None
            else:
                assert isinstance(subobj, types.FunctionType)
                func = subobj
            if inspect.ismethod(dynamic_subobj):
                return variables.UserMethodVariable(func, self, source=source)
            elif inspect.isfunction(dynamic_subobj):
                if is_utils_checkpoint(func):
                    return build_checkpoint_variable(source=source)
                elif is_allowed(func):
                    return variables.TorchVariable(func, source=source)
                return variables.UserFunctionVariable(func, source=source)
        if name in getattr(value, '__dict__', {}) or ConstantVariable.is_literal(subobj) or isinstance(subobj, (torch.Tensor, torch.nn.Module)):
            if source:
                return VariableBuilder(tx, source)(subobj)
            elif ConstantVariable.is_literal(subobj):
                return ConstantVariable.create(subobj)
        if name not in getattr(value, '__dict__', {}) and type(value).__module__.startswith('torch.') and ('torch.optim' not in type(value).__module__) and (not callable(value)):
            if not source:
                assert getattr(importlib.import_module(type(value).__module__), type(value).__name__) is type(value)
                source = AttrSource(AttrSource(tx.import_source(type(value).__module__), type(value).__name__), name)
            return VariableBuilder(tx, source)(subobj)
        options = {'source': source}
        if isinstance(subobj, (torch.distributions.constraints._Interval, torch.distributions.constraints._Real, torch.distributions.constraints.Constraint)):
            return UserDefinedObjectVariable(subobj, **options)
        elif isinstance(self.value, torch.nn.Module) and name in all_hook_names:
            assert isinstance(subobj, collections.OrderedDict)
            if not subobj:
                return variables.ConstDictVariable(subobj, collections.OrderedDict, **options)
        if name == '__class__':
            return UserDefinedClassVariable(type(self.value), **options)
        return variables.GetAttrVariable(self, name, **options)

    def call_hasattr(self, tx, name: str) -> 'VariableTracker':
        if tx.output.side_effects.is_attribute_mutation(self):
            try:
                result = tx.output.side_effects.load_attr(self, name, deleted_ok=True)
                return variables.ConstantVariable.create(not isinstance(result, variables.DeletedVariable))
            except KeyError:
                pass
        if self.source:
            install_guard(AttrSource(self.source, name).make_guard(GuardBuilder.HASATTR))
        if self._check_for_getattribute() or self._check_for_getattr():
            unimplemented('hasattr with custom __getattr__')
        try:
            self._getattr_static(name)
            return variables.ConstantVariable.create(True)
        except AttributeError:
            return variables.ConstantVariable.create(False)

    def odict_getitem(self, tx, key):
        from .builder import VariableBuilder
        index = key.source if ConstDictVariable.is_valid_key(key) and key.source is not None else key.as_python_constant()
        return VariableBuilder(tx, ODictGetItemSource(self.source, index))(collections.OrderedDict.__getitem__(self.value, key.as_python_constant()))