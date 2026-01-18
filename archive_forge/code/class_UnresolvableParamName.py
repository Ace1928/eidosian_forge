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
class UnresolvableParamName(ParamNameInterface, AbstractNameDefinition):

    def __init__(self, compiled_value, name, default):
        self.parent_context = compiled_value.parent_context
        self.string_name = name
        self._default = default

    def get_kind(self):
        return Parameter.POSITIONAL_ONLY

    def to_string(self):
        string = self.string_name
        if self._default:
            string += '=' + self._default
        return string

    def infer(self):
        return NO_VALUES