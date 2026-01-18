from abc import abstractmethod
from inspect import Parameter
from typing import Optional, Tuple
from parso.tree import search_ancestor
from jedi.parser_utils import find_statement_documentation, clean_scope_docstring
from jedi.inference.utils import unite
from jedi.inference.base_value import ValueSet, NO_VALUES
from jedi.inference.cache import inference_state_method_cache
from jedi.inference import docstrings
from jedi.cache import memoize_method
from jedi.inference.helpers import deep_ast_copy, infer_call_of_leaf
from jedi.plugins import plugin_manager
class AnonymousParamName(_ActualTreeParamName):

    @plugin_manager.decorate(name='goto_anonymous_param')
    def goto(self):
        return super().goto()

    @plugin_manager.decorate(name='infer_anonymous_param')
    def infer(self):
        values = super().infer()
        if values:
            return values
        from jedi.inference.dynamic_params import dynamic_param_lookup
        param = self._get_param_node()
        values = dynamic_param_lookup(self.function_value, param.position_index)
        if values:
            return values
        if param.star_count == 1:
            from jedi.inference.value.iterable import FakeTuple
            value = FakeTuple(self.function_value.inference_state, [])
        elif param.star_count == 2:
            from jedi.inference.value.iterable import FakeDict
            value = FakeDict(self.function_value.inference_state, {})
        elif param.default is None:
            return NO_VALUES
        else:
            return self.function_value.parent_context.infer_node(param.default)
        return ValueSet({value})