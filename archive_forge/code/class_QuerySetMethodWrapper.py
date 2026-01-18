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
class QuerySetMethodWrapper(ValueWrapper):

    def __init__(self, method, model_cls):
        super().__init__(method)
        self._model_cls = model_cls

    def py__get__(self, instance, class_value):
        return ValueSet({QuerySetBoundMethodWrapper(v, self._model_cls) for v in self._wrapped_value.py__get__(instance, class_value)})