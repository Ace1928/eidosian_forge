from jedi.inference.cache import inference_state_method_cache
from jedi.inference.base_value import ValueSet, NO_VALUES, Value, \
from jedi.inference.compiled import builtin_from_name
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.klass import ClassMixin
from jedi.inference.utils import to_list
from jedi.inference.names import AbstractNameDefinition, ValueName
from jedi.inference.context import ClassContext
from jedi.inference.gradual.generics import TupleGenericManager
class GenericClass(DefineGenericBaseClass, ClassMixin):
    """
    A class that is defined with generics, might be something simple like:

        class Foo(Generic[T]): ...
        my_foo_int_cls = Foo[int]
    """

    def __init__(self, class_value, generics_manager):
        super().__init__(generics_manager)
        self._class_value = class_value

    def _get_wrapped_value(self):
        return self._class_value

    def get_type_hint(self, add_class_info=True):
        n = self.py__name__()
        n = dict(list='List', dict='Dict', set='Set', tuple='Tuple').get(n, n)
        s = n + self._generics_manager.get_type_hint()
        if add_class_info:
            return 'Type[%s]' % s
        return s

    def get_type_var_filter(self):
        return _TypeVarFilter(self.get_generics(), self.list_type_vars())

    def py__call__(self, arguments):
        instance, = super().py__call__(arguments)
        return ValueSet([_GenericInstanceWrapper(instance)])

    def _as_context(self):
        return _AnnotatedClassContext(self)

    @to_list
    def py__bases__(self):
        for base in self._wrapped_value.py__bases__():
            yield _LazyGenericBaseClass(self, base, self._generics_manager)

    def _create_instance_with_generics(self, generics_manager):
        return GenericClass(self._class_value, generics_manager)

    def is_sub_class_of(self, class_value):
        if super().is_sub_class_of(class_value):
            return True
        return self._class_value.is_sub_class_of(class_value)

    def with_generics(self, generics_tuple):
        return self._class_value.with_generics(generics_tuple)

    def infer_type_vars(self, value_set):
        from jedi.inference.gradual.annotation import merge_pairwise_generics, merge_type_var_dicts
        annotation_name = self.py__name__()
        type_var_dict = {}
        if annotation_name == 'Iterable':
            annotation_generics = self.get_generics()
            if annotation_generics:
                return annotation_generics[0].infer_type_vars(value_set.merge_types_of_iterate())
        else:
            for py_class in value_set:
                if py_class.is_instance() and (not py_class.is_compiled()):
                    py_class = py_class.get_annotated_class_object()
                else:
                    continue
                if py_class.api_type != 'class':
                    continue
                for parent_class in py_class.py__mro__():
                    class_name = parent_class.py__name__()
                    if annotation_name == class_name:
                        merge_type_var_dicts(type_var_dict, merge_pairwise_generics(self, parent_class))
                        break
        return type_var_dict