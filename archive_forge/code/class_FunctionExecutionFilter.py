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
class FunctionExecutionFilter(_FunctionExecutionFilter):

    def __init__(self, *args, arguments, **kwargs):
        super().__init__(*args, **kwargs)
        self._arguments = arguments

    def _convert_param(self, param, name):
        return ParamName(self._function_value, name, self._arguments)