from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, \
from jedi.inference.compiled import builtin_from_name
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.klass import ClassMixin
from jedi.inference.utils import to_list
from jedi.inference.names import AbstractNameDefinition, ValueName
from jedi.inference.context import ClassContext
from jedi.inference.gradual.generics import TupleGenericManager
class BaseTypingInstance(LazyValueWrapper):

    def __init__(self, parent_context, class_value, tree_name, generics_manager):
        self.inference_state = class_value.inference_state
        self.parent_context = parent_context
        self._class_value = class_value
        self._tree_name = tree_name
        self._generics_manager = generics_manager

    def py__class__(self):
        return self._class_value

    def get_annotated_class_object(self):
        return self._class_value

    def get_qualified_names(self):
        return (self.py__name__(),)

    @property
    def name(self):
        return ValueName(self, self._tree_name)

    def _get_wrapped_value(self):
        object_, = builtin_from_name(self.inference_state, 'object').execute_annotation()
        return object_

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self._generics_manager)