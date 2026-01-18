import os
from pathlib import Path
from typing import Optional
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.names import AbstractNameDefinition, ModuleName
from jedi.inference.filters import GlobalNameFilter, ParserTreeFilter, DictFilter, MergedFilter
from jedi.inference import compiled
from jedi.inference.base_value import TreeValue
from jedi.inference.names import SubModuleName
from jedi.inference.helpers import values_from_qualified_names
from jedi.inference.compiled import create_simple_object
from jedi.inference.base_value import ValueSet
from jedi.inference.context import ModuleContext
class _ModuleAttributeName(AbstractNameDefinition):
    """
    For module attributes like __file__, __str__ and so on.
    """
    api_type = 'instance'

    def __init__(self, parent_module, string_name, string_value=None):
        self.parent_context = parent_module
        self.string_name = string_name
        self._string_value = string_value

    def infer(self):
        if self._string_value is not None:
            s = self._string_value
            return ValueSet([create_simple_object(self.parent_context.inference_state, s)])
        return compiled.get_string_value_set(self.parent_context.inference_state)