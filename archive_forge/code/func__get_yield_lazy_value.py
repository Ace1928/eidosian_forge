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
def _get_yield_lazy_value(self, yield_expr):
    if yield_expr.type == 'keyword':
        ctx = compiled.builtin_from_name(self.inference_state, 'None')
        yield LazyKnownValue(ctx)
        return
    node = yield_expr.children[1]
    if node.type == 'yield_arg':
        cn = ContextualizedNode(self, node.children[1])
        yield from cn.infer().iterate(cn)
    else:
        yield LazyTreeValue(self, node)