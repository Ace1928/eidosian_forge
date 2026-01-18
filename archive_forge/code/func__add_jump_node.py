import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def _add_jump_node(self, ast_node, guards):
    """Grows the graph by adding a jump node.

    Jump nodes are added to the current leaf set, and the leaf set becomes
    empty. If the jump node is the last in a cond section, then it may be added
    back to the leaf set by a separate mechanism.

    Args:
      ast_node: ast.AST
      guards: Tuple[ast.AST, ...], the finally sections active for this node

    Returns:
      Node
    """
    node = self._add_new_node(ast_node)
    self.leaves = set()
    self.finally_sections[node] = guards
    return node