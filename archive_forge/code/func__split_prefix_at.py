from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _split_prefix_at(leaf, until_line):
    """
    Returns a tuple of the leaf's prefix, split at the until_line
    position.
    """
    second_line_count = leaf.start_pos[0] - until_line
    lines = split_lines(leaf.prefix, keepends=True)
    return (''.join(lines[:-second_line_count]), ''.join(lines[-second_line_count:]))