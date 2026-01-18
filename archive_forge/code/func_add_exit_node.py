import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def add_exit_node(self, ast_node, section_id, guards):
    """Grows the graph by adding an exit node.

    This node becomes an exit for the current section.

    Args:
      ast_node: ast.AST
      section_id: Hashable, the node for which ast_node should be considered to
        be an exit node
      guards: Tuple[ast.AST, ...], the finally sections that guard ast_node

    Returns:
      Node
    """
    node = self._add_jump_node(ast_node, guards)
    self.exits[section_id].add(node)
    return node