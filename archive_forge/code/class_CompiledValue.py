import re
from functools import partial
from inspect import Parameter
from pathlib import Path
from typing import Optional
from jedi import debug
from jedi.inference.utils import to_list
from jedi.cache import memoize_method
from jedi.inference.filters import AbstractFilter
from jedi.inference.names import AbstractNameDefinition, ValueNameMixin, \
from jedi.inference.base_value import Value, ValueSet, NO_VALUES
from jedi.inference.lazy_value import LazyKnownValue
from jedi.inference.compiled.access import _sentinel
from jedi.inference.cache import inference_state_function_cache
from jedi.inference.helpers import reraise_getitem_errors
from jedi.inference.signature import BuiltinSignature
from jedi.inference.context import CompiledContext, CompiledModuleContext
class CompiledValue(Value):

    def __init__(self, inference_state, access_handle, parent_context=None):
        super().__init__(inference_state, parent_context)
        self.access_handle = access_handle

    def py__call__(self, arguments):
        return_annotation = self.access_handle.get_return_annotation()
        if return_annotation is not None:
            return create_from_access_path(self.inference_state, return_annotation).execute_annotation()
        try:
            self.access_handle.getattr_paths('__call__')
        except AttributeError:
            return super().py__call__(arguments)
        else:
            if self.access_handle.is_class():
                from jedi.inference.value import CompiledInstance
                return ValueSet([CompiledInstance(self.inference_state, self.parent_context, self, arguments)])
            else:
                return ValueSet(self._execute_function(arguments))

    @CheckAttribute()
    def py__class__(self):
        return create_from_access_path(self.inference_state, self.access_handle.py__class__())

    @CheckAttribute()
    def py__mro__(self):
        return (self,) + tuple((create_from_access_path(self.inference_state, access) for access in self.access_handle.py__mro__accesses()))

    @CheckAttribute()
    def py__bases__(self):
        return tuple((create_from_access_path(self.inference_state, access) for access in self.access_handle.py__bases__()))

    def get_qualified_names(self):
        return self.access_handle.get_qualified_names()

    def py__bool__(self):
        return self.access_handle.py__bool__()

    def is_class(self):
        return self.access_handle.is_class()

    def is_function(self):
        return self.access_handle.is_function()

    def is_module(self):
        return self.access_handle.is_module()

    def is_compiled(self):
        return True

    def is_stub(self):
        return False

    def is_instance(self):
        return self.access_handle.is_instance()

    def py__doc__(self):
        return self.access_handle.py__doc__()

    @to_list
    def get_param_names(self):
        try:
            signature_params = self.access_handle.get_signature_params()
        except ValueError:
            params_str, ret = self._parse_function_doc()
            if not params_str:
                tokens = []
            else:
                tokens = params_str.split(',')
            if self.access_handle.ismethoddescriptor():
                tokens.insert(0, 'self')
            for p in tokens:
                name, _, default = p.strip().partition('=')
                yield UnresolvableParamName(self, name, default)
        else:
            for signature_param in signature_params:
                yield SignatureParamName(self, signature_param)

    def get_signatures(self):
        _, return_string = self._parse_function_doc()
        return [BuiltinSignature(self, return_string)]

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.access_handle.get_repr())

    @memoize_method
    def _parse_function_doc(self):
        doc = self.py__doc__()
        if doc is None:
            return ('', '')
        return _parse_function_doc(doc)

    @property
    def api_type(self):
        return self.access_handle.get_api_type()

    def get_filters(self, is_instance=False, origin_scope=None):
        yield self._ensure_one_filter(is_instance)

    @memoize_method
    def _ensure_one_filter(self, is_instance):
        return CompiledValueFilter(self.inference_state, self, is_instance)

    def py__simple_getitem__(self, index):
        with reraise_getitem_errors(IndexError, KeyError, TypeError):
            try:
                access = self.access_handle.py__simple_getitem__(index, safe=not self.inference_state.allow_unsafe_executions)
            except AttributeError:
                return super().py__simple_getitem__(index)
        if access is None:
            return super().py__simple_getitem__(index)
        return ValueSet([create_from_access_path(self.inference_state, access)])

    def py__getitem__(self, index_value_set, contextualized_node):
        all_access_paths = self.access_handle.py__getitem__all_values()
        if all_access_paths is None:
            return super().py__getitem__(index_value_set, contextualized_node)
        return ValueSet((create_from_access_path(self.inference_state, access) for access in all_access_paths))

    def py__iter__(self, contextualized_node=None):
        if not self.access_handle.has_iter():
            yield from super().py__iter__(contextualized_node)
        access_path_list = self.access_handle.py__iter__list()
        if access_path_list is None:
            return
        for access in access_path_list:
            yield LazyKnownValue(create_from_access_path(self.inference_state, access))

    def py__name__(self):
        return self.access_handle.py__name__()

    @property
    def name(self):
        name = self.py__name__()
        if name is None:
            name = self.access_handle.get_repr()
        return CompiledValueName(self, name)

    def _execute_function(self, params):
        from jedi.inference import docstrings
        from jedi.inference.compiled import builtin_from_name
        if self.api_type != 'function':
            return
        for name in self._parse_function_doc()[1].split():
            try:
                self.inference_state.builtins_module.access_handle.getattr_paths(name)
            except AttributeError:
                continue
            else:
                bltn_obj = builtin_from_name(self.inference_state, name)
                yield from self.inference_state.execute(bltn_obj, params)
        yield from docstrings.infer_return_types(self)

    def get_safe_value(self, default=_sentinel):
        try:
            return self.access_handle.get_safe_value()
        except ValueError:
            if default == _sentinel:
                raise
            return default

    def execute_operation(self, other, operator):
        try:
            return ValueSet([create_from_access_path(self.inference_state, self.access_handle.execute_operation(other.access_handle, operator))])
        except TypeError:
            return NO_VALUES

    def execute_annotation(self):
        if self.access_handle.get_repr() == 'None':
            return ValueSet([self])
        name, args = self.access_handle.get_annotation_name_and_args()
        arguments = [ValueSet([create_from_access_path(self.inference_state, path)]) for path in args]
        if name == 'Union':
            return ValueSet.from_sets((arg.execute_annotation() for arg in arguments))
        elif name:
            return ValueSet([v.with_generics(arguments) for v in self.inference_state.typing_module.py__getattribute__(name)]).execute_annotation()
        return super().execute_annotation()

    def negate(self):
        return create_from_access_path(self.inference_state, self.access_handle.negate())

    def get_metaclasses(self):
        return NO_VALUES

    def _as_context(self):
        return CompiledContext(self)

    @property
    def array_type(self):
        return self.access_handle.get_array_type()

    def get_key_values(self):
        return [create_from_access_path(self.inference_state, k) for k in self.access_handle.get_key_paths()]

    def get_type_hint(self, add_class_info=True):
        if self.access_handle.get_repr() in ('None', "<class 'NoneType'>"):
            return 'None'
        return None