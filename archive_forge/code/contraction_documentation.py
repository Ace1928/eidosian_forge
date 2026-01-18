from itertools import chain, combinations, permutations, product
import networkx as nx
from networkx import density
from networkx.exception import NetworkXException
from networkx.utils import arbitrary_element
Returns the graph that results from contracting the specified edge.

    Edge contraction identifies the two endpoints of the edge as a single node
    incident to any edge that was incident to the original two nodes. A graph
    that results from edge contraction is called a *minor* of the original
    graph.

    Parameters
    ----------
    G : NetworkX graph
       The graph whose edge will be contracted.

    edge : tuple
       Must be a pair of nodes in `G`.

    self_loops : Boolean
       If this is True, any edges (including `edge`) joining the
       endpoints of `edge` in `G` become self-loops on the new node in the
       returned graph.

    copy : Boolean (default True)
        If this is True, a the contraction will be performed on a copy of `G`,
        otherwise the contraction will happen in place.

    Returns
    -------
    Networkx graph
       A new graph object of the same type as `G` (leaving `G` unmodified)
       with endpoints of `edge` identified in a single node. The right node
       of `edge` will be merged into the left one, so only the left one will
       appear in the returned graph.

    Raises
    ------
    ValueError
       If `edge` is not an edge in `G`.

    Examples
    --------
    Attempting to contract two nonadjacent nodes yields an error:

    >>> G = nx.cycle_graph(4)
    >>> nx.contracted_edge(G, (1, 3))
    Traceback (most recent call last):
      ...
    ValueError: Edge (1, 3) does not exist in graph G; cannot contract it

    Contracting two adjacent nodes in the cycle graph on *n* nodes yields the
    cycle graph on *n - 1* nodes:

    >>> C5 = nx.cycle_graph(5)
    >>> C4 = nx.cycle_graph(4)
    >>> M = nx.contracted_edge(C5, (0, 1), self_loops=False)
    >>> nx.is_isomorphic(M, C4)
    True

    See also
    --------
    contracted_nodes
    quotient_graph

    