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
class ProxyWithGenerics(BaseTypingClassWithGenerics):

    def execute_annotation(self):
        string_name = self._tree_name.value
        if string_name == 'Union':
            return self.gather_annotation_classes().execute_annotation()
        elif string_name == 'Optional':
            return self.gather_annotation_classes().execute_annotation() | ValueSet([builtin_from_name(self.inference_state, 'None')])
        elif string_name == 'Type':
            return self._generics_manager[0]
        elif string_name in ['ClassVar', 'Annotated']:
            return self._generics_manager[0].execute_annotation()
        mapped = {'Tuple': Tuple, 'Generic': Generic, 'Protocol': Protocol, 'Callable': Callable}
        cls = mapped[string_name]
        return ValueSet([cls(self.parent_context, self, self._tree_name, generics_manager=self._generics_manager)])

    def gather_annotation_classes(self):
        return ValueSet.from_sets(self._generics_manager.to_tuple())

    def _create_instance_with_generics(self, generics_manager):
        return ProxyWithGenerics(self.parent_context, self._tree_name, generics_manager)

    def infer_type_vars(self, value_set):
        annotation_generics = self.get_generics()
        if not annotation_generics:
            return {}
        annotation_name = self.py__name__()
        if annotation_name == 'Optional':
            none = builtin_from_name(self.inference_state, 'None')
            return annotation_generics[0].infer_type_vars(value_set.filter(lambda x: x != none))
        return {}