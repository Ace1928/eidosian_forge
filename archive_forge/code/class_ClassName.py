from jedi import debug
from jedi.parser_utils import get_cached_parent_scope, expr_is_dotted, \
from jedi.inference.cache import inference_state_method_cache, CachedMetaClass, \
from jedi.inference import compiled
from jedi.inference.lazy_value import LazyKnownValues, LazyTreeValue
from jedi.inference.filters import ParserTreeFilter
from jedi.inference.names import TreeNameDefinition, ValueName
from jedi.inference.arguments import unpack_arglist, ValuesArguments
from jedi.inference.base_value import ValueSet, iterator_to_value_set, \
from jedi.inference.context import ClassContext
from jedi.inference.value.function import FunctionAndClassBase
from jedi.inference.gradual.generics import LazyGenericManager, TupleGenericManager
from jedi.plugins import plugin_manager
class ClassName(TreeNameDefinition):

    def __init__(self, class_value, tree_name, name_context, apply_decorators):
        super().__init__(name_context, tree_name)
        self._apply_decorators = apply_decorators
        self._class_value = class_value

    @iterator_to_value_set
    def infer(self):
        from jedi.inference.syntax_tree import tree_name_to_values
        inferred = tree_name_to_values(self.parent_context.inference_state, self.parent_context, self.tree_name)
        for result_value in inferred:
            if self._apply_decorators:
                yield from result_value.py__get__(instance=None, class_value=self._class_value)
            else:
                yield result_value

    @property
    def api_type(self):
        type_ = super().api_type
        if type_ == 'function':
            definition = self.tree_name.get_definition()
            if definition is None:
                return type_
            if function_is_property(definition):
                return 'property'
        return type_