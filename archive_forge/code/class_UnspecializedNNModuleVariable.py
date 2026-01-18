import functools
import inspect
import itertools
import types
from contextlib import contextmanager, nullcontext
from typing import Dict, List
import torch.nn
from .. import skipfiles, variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented, UnspecializeRestartAnalysis, Unsupported
from ..guards import GuardBuilder, install_guard
from ..mutation_guard import GenerationTracker
from ..source import (
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .functions import invoke_and_store_as_constant
from .lists import SliceVariable
from .user_defined import UserDefinedObjectVariable
class UnspecializedNNModuleVariable(UserDefinedObjectVariable):
    _nonvar_fields = {'value_type', *UserDefinedObjectVariable._nonvar_fields}
    '\n    The above class will specialize on the id() of a module and place\n    parameters on the torch.fx.GraphModule.  Giving one graph per\n    module instance.  This version treats nn.Modules() like other user\n    defined objects and will pass parameters into the FX graph as inputs.\n    Giving one graph per module class.\n    '

    def __init__(self, value, **kwargs):
        if type(value) is torch.jit._script.RecursiveScriptModule:
            raise Unsupported("ScriptModules aren't supported in UnspecializedNNModuleVariable becuase their .forward function isn't a static member of their type")
        if 'value_type' in kwargs:
            lazy_value_to_become = getattr(kwargs['value_type'], 'cls_to_become', None)
            if type(value) is lazy_value_to_become:
                kwargs['value_type'] = type(value)
        super().__init__(value=value, **kwargs)

    @staticmethod
    @functools.lru_cache(None)
    def _nn_module_method_ids():
        return {id(x.__code__) for x in torch.nn.Module.__dict__.values() if hasattr(x, '__code__')}

    def unpack_var_sequence(self, tx):
        from .builder import VariableBuilder
        try:
            fn = inspect.getattr_static(self.value_type, '__iter__')
        except AttributeError as e:
            raise NotImplementedError from e
        if fn in (torch.nn.ModuleList.__iter__, torch.nn.ParameterList.__iter__, torch.nn.Sequential.__iter__):
            assert self.source
            return [VariableBuilder(tx, source=GetItemSource(self.source, idx))(item) for idx, item in enumerate(self.value)]
        return super().unpack_var_sequence(tx)

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        mod = self.value
        if is_lazy_module(mod):
            if mod.cls_to_become is not None:
                self.value_type = mod.cls_to_become
            initialize_lazy_module(tx, mod, args, kwargs)
        name = '_call_impl'
        fn = getattr(self.value_type, name)
        if self.source:
            source = AttrSource(AttrSource(self.source, '__class__'), name)
        else:
            source = None
        ctx = record_nn_module_stack(str(id(mod)), self.source, tx, mod) if self.source else nullcontext()
        with ctx:
            return variables.UserFunctionVariable(fn, source=source).call_function(tx, [self] + list(args), kwargs)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from .builder import VariableBuilder
        if name in ['_call_impl', '_wrapped_call_impl']:
            fn = getattr(self.value_type, name)
            if self.source:
                source = AttrSource(AttrSource(self.source, '__class__'), name)
            else:
                source = None
            return variables.UserFunctionVariable(fn, source=source).call_function(tx, [self] + list(args), kwargs)
        if name not in getattr(self.value, '__dict__', {}):
            try:
                method = inspect.getattr_static(type(self.value), name)
            except AttributeError:
                method = None
            if method is torch.nn.Module.parameters:
                assert not args or kwargs
                if tx.output.side_effects.has_pending_mutation(self):
                    unimplemented('Module.parameters() with pending mutation')
                install_guard(self.source.make_guard(GuardBuilder.NN_MODULE_PARAM_NAMES))
                items = []
                for name, value in self.value.named_parameters():
                    items.append(VariableBuilder(tx, AttrSource(self.source, name))(value))
                return variables.ListIteratorVariable(items, mutable_local=MutableLocal())
            elif isinstance(method, staticmethod):
                source = AttrSource(AttrSource(AttrSource(self.source, '__class__'), name), '__func__')
                return tx.inline_user_function_return(variables.UserFunctionVariable(method.__func__, source=source), args, kwargs)
            if id(method.__code__) in self._nn_module_method_ids():
                unimplemented(f'UnspecializedNNModuleVariable missing {name}')
        return super().call_method(tx, name, args, kwargs)