from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _find_non_global_names(nodes):
    for node in nodes:
        try:
            children = node.children
        except AttributeError:
            if node.type == 'name':
                yield node
        else:
            if node.type == 'trailer' and node.children[0] == '.':
                continue
            yield from _find_non_global_names(children)