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
class CompiledName(AbstractNameDefinition):

    def __init__(self, inference_state, parent_value, name, is_descriptor):
        self._inference_state = inference_state
        self.parent_context = parent_value.as_context()
        self._parent_value = parent_value
        self.string_name = name
        self.is_descriptor = is_descriptor

    def py__doc__(self):
        return self.infer_compiled_value().py__doc__()

    def _get_qualified_names(self):
        parent_qualified_names = self.parent_context.get_qualified_names()
        if parent_qualified_names is None:
            return None
        return parent_qualified_names + (self.string_name,)

    def get_defining_qualified_value(self):
        context = self.parent_context
        if context.is_module() or context.is_class():
            return self.parent_context.get_value()
        return None

    def __repr__(self):
        try:
            name = self.parent_context.name
        except AttributeError:
            name = None
        return '<%s: (%s).%s>' % (self.__class__.__name__, name, self.string_name)

    @property
    def api_type(self):
        if self.is_descriptor:
            return 'instance'
        return self.infer_compiled_value().api_type

    def infer(self):
        return ValueSet([self.infer_compiled_value()])

    @memoize_method
    def infer_compiled_value(self):
        return create_from_name(self._inference_state, self._parent_value, self.string_name)