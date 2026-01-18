from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _remove_unwanted_expression_nodes(parent_node, pos, until_pos):
    """
    This function makes it so for `1 * 2 + 3` you can extract `2 + 3`, even
    though it is not part of the expression.
    """
    typ = parent_node.type
    is_suite_part = typ in ('suite', 'file_input')
    if typ in EXPRESSION_PARTS or is_suite_part:
        nodes = parent_node.children
        for i, n in enumerate(nodes):
            if n.end_pos > pos:
                start_index = i
                if n.type == 'operator':
                    start_index -= 1
                break
        for i, n in reversed(list(enumerate(nodes))):
            if n.start_pos < until_pos:
                end_index = i
                if n.type == 'operator':
                    end_index += 1
                for n2 in nodes[i:]:
                    if _is_not_extractable_syntax(n2):
                        end_index += 1
                    else:
                        break
                break
        nodes = nodes[start_index:end_index + 1]
        if not is_suite_part:
            nodes[0:1] = _remove_unwanted_expression_nodes(nodes[0], pos, until_pos)
            nodes[-1:] = _remove_unwanted_expression_nodes(nodes[-1], pos, until_pos)
        return nodes
    return [parent_node]