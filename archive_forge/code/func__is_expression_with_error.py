from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _is_expression_with_error(nodes):
    """
    Returns a tuple (is_expression, error_string).
    """
    if any((node.type == 'name' and node.is_definition() for node in nodes)):
        return (False, 'Cannot extract a name that defines something')
    if nodes[0].type not in _VARIABLE_EXCTRACTABLE:
        return (False, 'Cannot extract a "%s"' % nodes[0].type)
    return (True, '')