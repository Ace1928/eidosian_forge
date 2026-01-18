import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def _add_new_node(self, ast_node):
    """Grows the graph by adding a CFG node following the current leaves."""
    if ast_node in self.node_index:
        raise ValueError('%s added twice' % ast_node)
    node = Node(next_=set(), prev=weakref.WeakSet(), ast_node=ast_node)
    self.node_index[ast_node] = node
    self.owners[node] = frozenset(self.active_stmts)
    if self.head is None:
        self.head = node
    for leaf in self.leaves:
        self._connect_nodes(leaf, node)
    for section_id in self.pending_finally_sections:
        self.finally_section_subgraphs[section_id][0] = node
    self.pending_finally_sections = set()
    return node