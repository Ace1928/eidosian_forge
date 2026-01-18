import re
from textwrap import dedent
from inspect import Parameter
from parso.python.token import PythonTokenTypes
from parso.python import tree
from parso.tree import search_ancestor, Leaf
from parso import split_lines
from jedi import debug
from jedi import settings
from jedi.api import classes
from jedi.api import helpers
from jedi.api import keywords
from jedi.api.strings import complete_dict
from jedi.api.file_name import complete_file_name
from jedi.inference import imports
from jedi.inference.base_value import ValueSet
from jedi.inference.helpers import infer_call_of_leaf, parse_dotted_names
from jedi.inference.context import get_global_filters
from jedi.inference.value import TreeInstance
from jedi.inference.docstring_utils import DocstringModule
from jedi.inference.names import ParamNameWrapper, SubModuleName
from jedi.inference.gradual.conversion import convert_values, convert_names
from jedi.parser_utils import cut_value_at_position
from jedi.plugins import plugin_manager
def _complete_params(self, leaf):
    stack_node = self.stack[-2]
    if stack_node.nonterminal == 'parameters':
        stack_node = self.stack[-3]
    if stack_node.nonterminal == 'funcdef':
        context = get_user_context(self._module_context, self._position)
        node = search_ancestor(leaf, 'error_node', 'funcdef')
        if node is not None:
            if node.type == 'error_node':
                n = node.children[0]
                if n.type == 'decorators':
                    decorators = n.children
                elif n.type == 'decorator':
                    decorators = [n]
                else:
                    decorators = []
            else:
                decorators = node.get_decorators()
            function_name = stack_node.nodes[1]
            return complete_param_names(context, function_name.value, decorators)
    return []