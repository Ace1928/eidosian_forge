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
class _ParamMixin:

    def maybe_positional_argument(self, include_star=True):
        options = [Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD]
        if include_star:
            options.append(Parameter.VAR_POSITIONAL)
        return self.get_kind() in options

    def maybe_keyword_argument(self, include_stars=True):
        options = [Parameter.KEYWORD_ONLY, Parameter.POSITIONAL_OR_KEYWORD]
        if include_stars:
            options.append(Parameter.VAR_KEYWORD)
        return self.get_kind() in options

    def _kind_string(self):
        kind = self.get_kind()
        if kind == Parameter.VAR_POSITIONAL:
            return '*'
        if kind == Parameter.VAR_KEYWORD:
            return '**'
        return ''

    def get_qualified_names(self, include_module_names=False):
        return None