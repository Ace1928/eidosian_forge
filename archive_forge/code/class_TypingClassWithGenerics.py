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
class TypingClassWithGenerics(ProxyWithGenerics, _TypingClassMixin):

    def infer_type_vars(self, value_set):
        type_var_dict = {}
        annotation_generics = self.get_generics()
        if not annotation_generics:
            return type_var_dict
        annotation_name = self.py__name__()
        if annotation_name == 'Type':
            return annotation_generics[0].infer_type_vars(value_set.execute_annotation())
        elif annotation_name == 'Callable':
            if len(annotation_generics) == 2:
                return annotation_generics[1].infer_type_vars(value_set.execute_annotation())
        elif annotation_name == 'Tuple':
            tuple_annotation, = self.execute_annotation()
            return tuple_annotation.infer_type_vars(value_set)
        return type_var_dict

    def _create_instance_with_generics(self, generics_manager):
        return TypingClassWithGenerics(self.parent_context, self._tree_name, generics_manager)