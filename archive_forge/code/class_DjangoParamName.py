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
class DjangoParamName(BaseTreeParamName):

    def __init__(self, field_name):
        super().__init__(field_name.parent_context, field_name.tree_name)
        self._field_name = field_name

    def get_kind(self):
        return Parameter.KEYWORD_ONLY

    def infer(self):
        return self._field_name.infer()