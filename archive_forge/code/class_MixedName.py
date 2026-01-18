import inspect
from pathlib import Path
from jedi.parser_utils import get_cached_code_lines
from jedi import settings
from jedi.cache import memoize_method
from jedi.inference import compiled
from jedi.file_io import FileIO
from jedi.inference.names import NameWrapper
from jedi.inference.base_value import ValueSet, ValueWrapper, NO_VALUES
from jedi.inference.value import ModuleValue
from jedi.inference.cache import inference_state_function_cache, \
from jedi.inference.compiled.access import ALLOWED_GETITEM_TYPES, get_api_type
from jedi.inference.gradual.conversion import to_stub
from jedi.inference.context import CompiledContext, CompiledModuleContext, \
class MixedName(NameWrapper):
    """
    The ``CompiledName._compiled_value`` is our MixedObject.
    """

    def __init__(self, wrapped_name, parent_tree_value):
        super().__init__(wrapped_name)
        self._parent_tree_value = parent_tree_value

    @property
    def start_pos(self):
        values = list(self.infer())
        if not values:
            return (0, 0)
        return values[0].name.start_pos

    @memoize_method
    def infer(self):
        compiled_value = self._wrapped_name.infer_compiled_value()
        tree_value = self._parent_tree_value
        if tree_value.is_instance() or tree_value.is_class():
            tree_values = tree_value.py__getattribute__(self.string_name)
            if compiled_value.is_function():
                return ValueSet({MixedObject(compiled_value, v) for v in tree_values})
        module_context = tree_value.get_root_context()
        return _create(self._inference_state, compiled_value, module_context)