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
class SubModuleDictMixin:

    @inference_state_method_cache()
    def sub_modules_dict(self):
        """
        Lists modules in the directory of this module (if this module is a
        package).
        """
        names = {}
        if self.is_package():
            mods = self.inference_state.compiled_subprocess.iter_module_names(self.py__path__())
            for name in mods:
                names[name] = SubModuleName(self.as_context(), name)
        return names