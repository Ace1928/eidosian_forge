from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _is_node_ending_return_stmt(node):
    t = node.type
    if t == 'simple_stmt':
        return _is_node_ending_return_stmt(node.children[0])
    return t == 'return_stmt'