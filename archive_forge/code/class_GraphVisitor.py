import collections
import enum
import weakref
import astunparse
import gast
from tensorflow.python.autograph.pyct import anno
class GraphVisitor(object):
    """Base class for a CFG visitors.

  This implementation is not thread safe.

  The visitor has some facilities to simplify dataflow analyses. In particular,
  it allows revisiting the nodes at the decision of the subclass. This can be
  used to visit the graph until the state reaches a fixed point.

  For more details on dataflow analysis, see
  https://www.seas.harvard.edu/courses/cs252/2011sp/slides/Lec02-Dataflow.pdf

  Note: the literature generally suggests visiting successor nodes only when the
  state of the current node changed, regardless of whether that successor has
  ever been visited. This implementation visits every successor at least once.

  Attributes:
    graph: Graph
    in_: Dict[Node, Any], stores node-keyed state during a visit
    out: Dict[Node, Any], stores node-keyed state during a visit
  """

    def __init__(self, graph):
        self.graph = graph
        self.reset()

    def init_state(self, node):
        """State initialization function.

    Optional to overload.

    An in/out state slot will be created for each node in the graph. Subclasses
    must overload this to control what that is initialized to.

    Args:
      node: Node
    """
        raise NotImplementedError('Subclasses must implement this.')

    def visit_node(self, node):
        """Visitor function.

    Args:
      node: Node

    Returns:
      bool, whether the node should be revisited; subclasses can visit every
          reachable node exactly once by always returning False
    """
        raise NotImplementedError('Subclasses must implement this.')

    def reset(self):
        self.in_ = {node: self.init_state(node) for node in self.graph.index.values()}
        self.out = {node: self.init_state(node) for node in self.graph.index.values()}

    def can_ignore(self, node):
        """Returns True if the node can safely be assumed not to touch variables."""
        ast_node = node.ast_node
        if anno.hasanno(ast_node, anno.Basic.SKIP_PROCESSING):
            return True
        return isinstance(ast_node, (gast.Break, gast.Continue, gast.Raise, gast.Pass))

    def _visit_internal(self, mode):
        """Visits the CFG, breadth-first."""
        assert mode in (_WalkMode.FORWARD, _WalkMode.REVERSE)
        if mode == _WalkMode.FORWARD:
            open_ = [self.graph.entry]
        elif mode == _WalkMode.REVERSE:
            open_ = list(self.graph.exit)
        closed = set()
        while open_:
            node = open_.pop(0)
            closed.add(node)
            should_revisit = self.visit_node(node)
            if mode == _WalkMode.FORWARD:
                children = node.next
            elif mode == _WalkMode.REVERSE:
                children = node.prev
            for next_ in children:
                if should_revisit or next_ not in closed:
                    open_.append(next_)

    def visit_forward(self):
        self._visit_internal(_WalkMode.FORWARD)

    def visit_reverse(self):
        self._visit_internal(_WalkMode.REVERSE)