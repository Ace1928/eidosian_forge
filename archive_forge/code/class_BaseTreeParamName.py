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
class BaseTreeParamName(ParamNameInterface, AbstractTreeName):
    annotation_node = None
    default_node = None

    def to_string(self):
        output = self._kind_string() + self.get_public_name()
        annotation = self.annotation_node
        default = self.default_node
        if annotation is not None:
            output += ': ' + annotation.get_code(include_prefix=False)
        if default is not None:
            output += '=' + default.get_code(include_prefix=False)
        return output

    def get_public_name(self):
        name = self.string_name
        if name.startswith('__'):
            name = name[2:]
        return name

    def goto(self, **kwargs):
        return [self]