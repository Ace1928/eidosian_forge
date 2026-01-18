import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
def _connect_nodes(self, first, second):
    """Connects nodes to signify that control flows from first to second.

    Args:
      first: Union[Set[Node, ...], Node]
      second: Node
    """
    if isinstance(first, Node):
        first.next.add(second)
        second.prev.add(first)
        self.forward_edges.add((first, second))
    else:
        for node in first:
            self._connect_nodes(node, second)