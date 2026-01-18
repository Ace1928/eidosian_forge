from textwrap import dedent
from parso import split_lines
from jedi import debug
from jedi.api.exceptions import RefactoringError
from jedi.api.refactoring import Refactoring, EXPRESSION_PARTS
from jedi.common import indent_block
from jedi.parser_utils import function_is_classmethod, function_is_staticmethod
def _find_nodes(module_node, pos, until_pos):
    """
    Looks up a module and tries to find the appropriate amount of nodes that
    are in there.
    """
    start_node = module_node.get_leaf_for_position(pos, include_prefixes=True)
    if until_pos is None:
        if start_node.type == 'operator':
            next_leaf = start_node.get_next_leaf()
            if next_leaf is not None and next_leaf.start_pos == pos:
                start_node = next_leaf
        if _is_not_extractable_syntax(start_node):
            start_node = start_node.parent
        if start_node.parent.type == 'trailer':
            start_node = start_node.parent.parent
        while start_node.parent.type in EXPRESSION_PARTS:
            start_node = start_node.parent
        nodes = [start_node]
    else:
        if start_node.end_pos == pos:
            next_leaf = start_node.get_next_leaf()
            if next_leaf is not None:
                start_node = next_leaf
        if _is_not_extractable_syntax(start_node):
            start_node = start_node.parent
        end_leaf = module_node.get_leaf_for_position(until_pos, include_prefixes=True)
        if end_leaf.start_pos > until_pos:
            end_leaf = end_leaf.get_previous_leaf()
            if end_leaf is None:
                raise RefactoringError('Cannot extract anything from that')
        parent_node = start_node
        while parent_node.end_pos < end_leaf.end_pos:
            parent_node = parent_node.parent
        nodes = _remove_unwanted_expression_nodes(parent_node, pos, until_pos)
    if len(nodes) == 1 and start_node.type in ('return_stmt', 'yield_expr'):
        return [nodes[0].children[1]]
    return nodes