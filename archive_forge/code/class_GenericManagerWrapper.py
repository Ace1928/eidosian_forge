from inspect import Parameter
from jedi import debug
from jedi.inference.cache import inference_state_function_cache
from jedi.inference.base_value import ValueSet, iterator_to_value_set, ValueWrapper
from jedi.inference.filters import DictFilter, AttributeOverwrite
from jedi.inference.names import NameWrapper, BaseTreeParamName
from jedi.inference.compiled.value import EmptyCompiledName
from jedi.inference.value.instance import TreeInstance
from jedi.inference.value.klass import ClassMixin
from jedi.inference.gradual.base import GenericClass
from jedi.inference.gradual.generics import TupleGenericManager
from jedi.inference.signature import AbstractSignature
class GenericManagerWrapper(AttributeOverwrite, ClassMixin):

    def py__get__on_class(self, calling_instance, instance, class_value):
        return calling_instance.class_value.with_generics((ValueSet({class_value}),)).py__call__(calling_instance._arguments)

    def with_generics(self, generics_tuple):
        return self._wrapped_value.with_generics(generics_tuple)