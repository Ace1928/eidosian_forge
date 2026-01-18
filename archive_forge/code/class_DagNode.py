from __future__ import unicode_literals
from ._utils import get_hash, get_hash_int
from builtins import object
from collections import namedtuple
class DagNode(object):
    """Node in a directed-acyclic graph (DAG).

    Edges:
        DagNodes are connected by edges.  An edge connects two nodes with a label for each side:
         - ``upstream_node``: upstream/parent node
         - ``upstream_label``: label on the outgoing side of the upstream node
         - ``downstream_node``: downstream/child node
         - ``downstream_label``: label on the incoming side of the downstream node

        For example, DagNode A may be connected to DagNode B with an edge labelled "foo" on A's side, and "bar" on B's
        side:

           _____               _____
          |     |             |     |
          |  A  >[foo]---[bar]>  B  |
          |_____|             |_____|

        Edge labels may be integers or strings, and nodes cannot have more than one incoming edge with the same label.

        DagNodes may have any number of incoming edges and any number of outgoing edges.  DagNodes keep track only of
        their incoming edges, but the entire graph structure can be inferred by looking at the furthest downstream
        nodes and working backwards.

    Hashing:
        DagNodes must be hashable, and two nodes are considered to be equivalent if they have the same hash value.

        Nodes are immutable, and the hash should remain constant as a result.  If a node with new contents is required,
        create a new node and throw the old one away.

    String representation:
        In order for graph visualization tools to show useful information, nodes must be representable as strings.  The
        ``repr`` operator should provide a more or less "full" representation of the node, and the ``short_repr``
        property should be a shortened, concise representation.

        Again, because nodes are immutable, the string representations should remain constant.
    """

    def __hash__(self):
        """Return an integer hash of the node."""
        raise NotImplementedError()

    def __eq__(self, other):
        """Compare two nodes; implementations should return True if (and only if) hashes match."""
        raise NotImplementedError()

    def __repr__(self, other):
        """Return a full string representation of the node."""
        raise NotImplementedError()

    @property
    def short_repr(self):
        """Return a partial/concise representation of the node."""
        raise NotImplementedError()

    @property
    def incoming_edge_map(self):
        """Provides information about all incoming edges that connect to this node.

        The edge map is a dictionary that maps an ``incoming_label`` to ``(outgoing_node, outgoing_label)``.  Note that
        implicity, ``incoming_node`` is ``self``.  See "Edges" section above.
        """
        raise NotImplementedError()