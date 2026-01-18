from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, \
from jedi.inference.compiled import builtin_from_name
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.klass import ClassMixin
from jedi.inference.utils import to_list
from jedi.inference.names import AbstractNameDefinition, ValueName
from jedi.inference.context import ClassContext
from jedi.inference.gradual.generics import TupleGenericManager
class DefineGenericBaseClass(LazyValueWrapper):

    def __init__(self, generics_manager):
        self._generics_manager = generics_manager

    def _create_instance_with_generics(self, generics_manager):
        raise NotImplementedError

    @inference_state_method_cache()
    def get_generics(self):
        return self._generics_manager.to_tuple()

    def define_generics(self, type_var_dict):
        from jedi.inference.gradual.type_var import TypeVar
        changed = False
        new_generics = []
        for generic_set in self.get_generics():
            values = NO_VALUES
            for generic in generic_set:
                if isinstance(generic, (DefineGenericBaseClass, TypeVar)):
                    result = generic.define_generics(type_var_dict)
                    values |= result
                    if result != ValueSet({generic}):
                        changed = True
                else:
                    values |= ValueSet([generic])
            new_generics.append(values)
        if not changed:
            return ValueSet([self])
        return ValueSet([self._create_instance_with_generics(TupleGenericManager(tuple(new_generics)))])

    def is_same_class(self, other):
        if not isinstance(other, DefineGenericBaseClass):
            return False
        if self.tree_node != other.tree_node:
            return False
        given_params1 = self.get_generics()
        given_params2 = other.get_generics()
        if len(given_params1) != len(given_params2):
            return False
        return all((any((cls2.is_same_class(cls1) for cls1 in class_set1.gather_annotation_classes() for cls2 in class_set2.gather_annotation_classes())) for class_set1, class_set2 in zip(given_params1, given_params2)))

    def get_signatures(self):
        return []

    def __repr__(self):
        return '<%s: %s%s>' % (self.__class__.__name__, self._wrapped_value, list(self.get_generics()))