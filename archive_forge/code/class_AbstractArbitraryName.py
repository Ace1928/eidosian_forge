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
class AbstractArbitraryName(AbstractNameDefinition):
    """
    When you e.g. want to complete dicts keys, you probably want to complete
    string literals, which is not really a name, but for Jedi we use this
    concept of Name for completions as well.
    """
    is_value_name = False

    def __init__(self, inference_state, string):
        self.inference_state = inference_state
        self.string_name = string
        self.parent_context = inference_state.builtins_module

    def infer(self):
        return NO_VALUES