from parso.python import tree
from jedi import debug
from jedi.inference.cache import inference_state_method_cache, CachedMetaClass
from jedi.inference import compiled
from jedi.inference import recursion
from jedi.inference import docstrings
from jedi.inference import flow_analysis
from jedi.inference.signature import TreeSignature
from jedi.inference.filters import ParserTreeFilter, FunctionExecutionFilter, \
from jedi.inference.names import ValueName, AbstractNameDefinition, \
from jedi.inference.base_value import ContextualizedNode, NO_VALUES, \
from jedi.inference.lazy_value import LazyKnownValues, LazyKnownValue, \
from jedi.inference.context import ValueContext, TreeContextMixin
from jedi.inference.value import iterable
from jedi import parser_utils
from jedi.inference.parser_cache import get_yield_exprs
from jedi.inference.helpers import values_from_qualified_names
from jedi.inference.gradual.generics import TupleGenericManager
class OverloadedFunctionValue(FunctionMixin, ValueWrapper):

    def __init__(self, function, overloaded_functions):
        super().__init__(function)
        self._overloaded_functions = overloaded_functions

    def py__call__(self, arguments):
        debug.dbg('Execute overloaded function %s', self._wrapped_value, color='BLUE')
        function_executions = []
        for signature in self.get_signatures():
            function_execution = signature.value.as_context(arguments)
            function_executions.append(function_execution)
            if signature.matches_signature(arguments):
                return function_execution.infer()
        if self.inference_state.is_analysis:
            return NO_VALUES
        return ValueSet.from_sets((fe.infer() for fe in function_executions))

    def get_signature_functions(self):
        return self._overloaded_functions

    def get_type_hint(self, add_class_info=True):
        return 'Union[%s]' % ', '.join((f.get_type_hint() for f in self._overloaded_functions))