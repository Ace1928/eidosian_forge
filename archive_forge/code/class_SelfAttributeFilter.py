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
class SelfAttributeFilter(ClassFilter):
    """
    This class basically filters all the use cases where `self.*` was assigned.
    """

    def __init__(self, instance, instance_class, node_context, origin_scope):
        super().__init__(class_value=instance_class, node_context=node_context, origin_scope=origin_scope, is_instance=True)
        self._instance = instance

    def _filter(self, names):
        start, end = (self._parser_scope.start_pos, self._parser_scope.end_pos)
        names = [n for n in names if start < n.start_pos < end]
        return self._filter_self_names(names)

    def _filter_self_names(self, names):
        for name in names:
            trailer = name.parent
            if trailer.type == 'trailer' and len(trailer.parent.children) == 2 and (trailer.children[0] == '.'):
                if name.is_definition() and self._access_possible(name):
                    if self._is_in_right_scope(trailer.parent.children[0], name):
                        yield name

    def _is_in_right_scope(self, self_name, name):
        self_context = self._node_context.create_context(self_name)
        names = self_context.goto(self_name, position=self_name.start_pos)
        return any((n.api_type == 'param' and n.tree_name.get_definition().position_index == 0 and (n.parent_context.tree_node is self._parser_scope) for n in names))

    def _convert_names(self, names):
        return [SelfName(self._instance, self._node_context, name) for name in names]

    def _check_flows(self, names):
        return names