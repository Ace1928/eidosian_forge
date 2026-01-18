from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _is_not_extractable_syntax(node):
    return node.type == 'operator' or (node.type == 'keyword' and node.value not in ('None', 'True', 'False'))