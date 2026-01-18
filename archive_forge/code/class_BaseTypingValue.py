from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, \
from jedi.inference.compiled import builtin_from_name
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.klass import ClassMixin
from jedi.inference.utils import to_list
from jedi.inference.names import AbstractNameDefinition, ValueName
from jedi.inference.context import ClassContext
from jedi.inference.gradual.generics import TupleGenericManager
class BaseTypingValue(LazyValueWrapper):

    def __init__(self, parent_context, tree_name):
        self.inference_state = parent_context.inference_state
        self.parent_context = parent_context
        self._tree_name = tree_name

    @property
    def name(self):
        return ValueName(self, self._tree_name)

    def _get_wrapped_value(self):
        return _PseudoTreeNameClass(self.parent_context, self._tree_name)

    def get_signatures(self):
        return self._wrapped_value.get_signatures()

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self._tree_name.value)