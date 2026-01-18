from abc import abstractmethod
from typing import List, MutableMapping, Type
import weakref
from parso.tree import search_ancestor
from parso.python.tree import Name, UsedNamesMapping
from jedi.inference import flow_analysis
from jedi.inference.base_value import ValueSet, ValueWrapper, \
from jedi.parser_utils import get_cached_parent_scope, get_parso_cache_node
from jedi.inference.utils import to_list
from jedi.inference.names import TreeNameDefinition, ParamName, \
class _FunctionExecutionFilter(ParserTreeFilter):

    def __init__(self, parent_context, function_value, until_position, origin_scope):
        super().__init__(parent_context, until_position=until_position, origin_scope=origin_scope)
        self._function_value = function_value

    def _convert_param(self, param, name):
        raise NotImplementedError

    @to_list
    def _convert_names(self, names):
        for name in names:
            param = search_ancestor(name, 'param')
            if param:
                yield self._convert_param(param, name)
            else:
                yield TreeNameDefinition(self.parent_context, name)