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
class FunctionExecutionContext(BaseFunctionExecutionContext):

    def __init__(self, function_value, arguments):
        super().__init__(function_value)
        self._arguments = arguments

    def get_filters(self, until_position=None, origin_scope=None):
        yield FunctionExecutionFilter(self, self._value, until_position=until_position, origin_scope=origin_scope, arguments=self._arguments)

    def infer_annotations(self):
        from jedi.inference.gradual.annotation import infer_return_types
        return infer_return_types(self._value, self._arguments)

    def get_param_names(self):
        return [ParamName(self._value, param.name, self._arguments) for param in self._value.tree_node.get_params()]