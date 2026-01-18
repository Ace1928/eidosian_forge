from abc import abstractproperty
from parso.tree import search_ancestor
from jedi import debug
from jedi import settings
from jedi.inference import compiled
from jedi.inference.compiled.value import CompiledValueFilter
from jedi.inference.helpers import values_from_qualified_names, is_big_annoying_library
from jedi.inference.filters import AbstractFilter, AnonymousFunctionExecutionFilter
from jedi.inference.names import ValueName, TreeNameDefinition, ParamName, \
from jedi.inference.base_value import Value, NO_VALUES, ValueSet, \
from jedi.inference.lazy_value import LazyKnownValue, LazyKnownValues
from jedi.inference.cache import inference_state_method_cache
from jedi.inference.arguments import ValuesArguments, TreeArgumentsWrapper
from jedi.inference.value.function import \
from jedi.inference.value.klass import ClassFilter
from jedi.inference.value.dynamic_arrays import get_dynamic_array_instance
from jedi.parser_utils import function_is_staticmethod, function_is_classmethod
class AnonymousMethodExecutionContext(BaseFunctionExecutionContext):

    def __init__(self, instance, value):
        super().__init__(value)
        self.instance = instance

    def get_filters(self, until_position=None, origin_scope=None):
        yield AnonymousMethodExecutionFilter(self.instance, self, self._value, until_position=until_position, origin_scope=origin_scope)

    def get_param_names(self):
        param_names = list(self._value.get_param_names())
        param_names[0] = InstanceExecutedParamName(self.instance, self._value, param_names[0].tree_name)
        return param_names