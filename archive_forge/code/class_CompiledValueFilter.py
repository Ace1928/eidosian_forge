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
class CompiledValueFilter(AbstractFilter):

    def __init__(self, inference_state, compiled_value, is_instance=False):
        self._inference_state = inference_state
        self.compiled_value = compiled_value
        self.is_instance = is_instance

    def get(self, name):
        access_handle = self.compiled_value.access_handle
        safe = not self._inference_state.allow_unsafe_executions
        return self._get(name, lambda name: access_handle.is_allowed_getattr(name, safe=safe), lambda name: name in access_handle.dir(), check_has_attribute=True)

    def _get(self, name, allowed_getattr_callback, in_dir_callback, check_has_attribute=False):
        """
        To remove quite a few access calls we introduced the callback here.
        """
        has_attribute, is_descriptor, property_return_annotation = allowed_getattr_callback(name)
        if property_return_annotation is not None:
            values = create_from_access_path(self._inference_state, property_return_annotation).execute_annotation()
            if values:
                return [CompiledValueName(v, name) for v in values]
        if check_has_attribute and (not has_attribute):
            return []
        if (is_descriptor or not has_attribute) and (not self._inference_state.allow_unsafe_executions):
            return [self._get_cached_name(name, is_empty=True)]
        if self.is_instance and (not in_dir_callback(name)):
            return []
        return [self._get_cached_name(name, is_descriptor=is_descriptor)]

    @memoize_method
    def _get_cached_name(self, name, is_empty=False, *, is_descriptor=False):
        if is_empty:
            return EmptyCompiledName(self._inference_state, name)
        else:
            return self._create_name(name, is_descriptor=is_descriptor)

    def values(self):
        from jedi.inference.compiled import builtin_from_name
        names = []
        needs_type_completions, dir_infos = self.compiled_value.access_handle.get_dir_infos()
        for name in dir_infos:
            names += self._get(name, lambda name: dir_infos[name], lambda name: name in dir_infos)
        if not self.is_instance and needs_type_completions:
            for filter in builtin_from_name(self._inference_state, 'type').get_filters():
                names += filter.values()
        return names

    def _create_name(self, name, is_descriptor):
        return CompiledName(self._inference_state, self.compiled_value, name, is_descriptor)

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self.compiled_value)