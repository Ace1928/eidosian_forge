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
@inference_state_method_cache()
def _module_attributes_dict(self):
    names = ['__package__', '__doc__', '__name__']
    dct = dict(((n, _ModuleAttributeName(self, n)) for n in names))
    path = self.py__file__()
    if path is not None:
        dct['__file__'] = _ModuleAttributeName(self, '__file__', str(path))
    return dct