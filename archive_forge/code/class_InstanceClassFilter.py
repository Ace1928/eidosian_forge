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
class InstanceClassFilter(AbstractFilter):
    """
    This filter is special in that it uses the class filter and wraps the
    resulting names in LazyInstanceClassName. The idea is that the class name
    filtering can be very flexible and always be reflected in instances.
    """

    def __init__(self, instance, class_filter):
        self._instance = instance
        self._class_filter = class_filter

    def get(self, name):
        return self._convert(self._class_filter.get(name))

    def values(self):
        return self._convert(self._class_filter.values())

    def _convert(self, names):
        return [LazyInstanceClassName(self._instance, n) for n in names]

    def __repr__(self):
        return '<%s for %s>' % (self.__class__.__name__, self._class_filter)