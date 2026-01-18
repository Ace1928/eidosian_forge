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
class SpecialMethodName(AbstractNameDefinition):
    api_type = 'function'

    def __init__(self, parent_context, string_name, callable_, builtin_value):
        self.parent_context = parent_context
        self.string_name = string_name
        self._callable = callable_
        self._builtin_value = builtin_value

    def infer(self):
        for filter in self._builtin_value.get_filters():
            for name in filter.get(self.string_name):
                builtin_func = next(iter(name.infer()))
                break
            else:
                continue
            break
        return ValueSet([_BuiltinMappedMethod(self.parent_context, self._callable, builtin_func)])