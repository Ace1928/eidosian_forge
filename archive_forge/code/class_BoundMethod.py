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
class BoundMethod(FunctionMixin, ValueWrapper):

    def __init__(self, instance, class_context, function):
        super().__init__(function)
        self.instance = instance
        self._class_context = class_context

    def is_bound_method(self):
        return True

    @property
    def name(self):
        return FunctionNameInClass(self._class_context, super().name)

    def py__class__(self):
        c, = values_from_qualified_names(self.inference_state, 'types', 'MethodType')
        return c

    def _get_arguments(self, arguments):
        assert arguments is not None
        return InstanceArguments(self.instance, arguments)

    def _as_context(self, arguments=None):
        if arguments is None:
            return AnonymousMethodExecutionContext(self.instance, self)
        arguments = self._get_arguments(arguments)
        return MethodExecutionContext(self.instance, self, arguments)

    def py__call__(self, arguments):
        if isinstance(self._wrapped_value, OverloadedFunctionValue):
            return self._wrapped_value.py__call__(self._get_arguments(arguments))
        function_execution = self.as_context(arguments)
        return function_execution.infer()

    def get_signature_functions(self):
        return [BoundMethod(self.instance, self._class_context, f) for f in self._wrapped_value.get_signature_functions()]

    def get_signatures(self):
        return [sig.bind(self) for sig in super().get_signatures()]

    def __repr__(self):
        return '<%s: %s>' % (self.__class__.__name__, self._wrapped_value)