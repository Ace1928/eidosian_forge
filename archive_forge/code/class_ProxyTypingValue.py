import itertools
from jedi import debug
from jedi.inference.compiled import builtin_from_name, create_simple_object
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, \
from jedi.inference.lazy_value import LazyKnownValues
from jedi.inference.arguments import repack_with_argument_clinic
from jedi.inference.filters import FilterWrapper
from jedi.inference.names import NameWrapper, ValueName
from jedi.inference.value.klass import ClassMixin
from jedi.inference.gradual.base import BaseTypingValue, \
from jedi.inference.gradual.type_var import TypeVarClass
from jedi.inference.gradual.generics import LazyGenericManager, TupleGenericManager
class ProxyTypingValue(BaseTypingValue):
    index_class = ProxyWithGenerics

    def with_generics(self, generics_tuple):
        return self.index_class.create_cached(self.inference_state, self.parent_context, self._tree_name, generics_manager=TupleGenericManager(generics_tuple))

    def py__getitem__(self, index_value_set, contextualized_node):
        return ValueSet((self.index_class.create_cached(self.inference_state, self.parent_context, self._tree_name, generics_manager=LazyGenericManager(context_of_index=contextualized_node.context, index_value=index_value)) for index_value in index_value_set))