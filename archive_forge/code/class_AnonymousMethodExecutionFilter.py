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
class AnonymousMethodExecutionFilter(AnonymousFunctionExecutionFilter):

    def __init__(self, instance, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._instance = instance

    def _convert_param(self, param, name):
        if param.position_index == 0:
            if function_is_classmethod(self._function_value.tree_node):
                return InstanceExecutedParamName(self._instance.py__class__(), self._function_value, name)
            elif not function_is_staticmethod(self._function_value.tree_node):
                return InstanceExecutedParamName(self._instance, self._function_value, name)
        return super()._convert_param(param, name)