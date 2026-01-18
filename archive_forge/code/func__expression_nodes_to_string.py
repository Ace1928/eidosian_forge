from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _expression_nodes_to_string(nodes):
    return ''.join((n.get_code(include_prefix=i != 0) for i, n in enumerate(nodes)))