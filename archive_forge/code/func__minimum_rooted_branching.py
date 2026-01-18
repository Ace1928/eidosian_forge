import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
def _minimum_rooted_branching(D, root):
    """Helper function to compute a minimum rooted branching (aka rooted
    arborescence)

    Before the branching can be computed, the directed graph must be rooted by
    removing the predecessors of root.

    A branching / arborescence of rooted graph G is a subgraph that contains a
    directed path from the root to every other vertex. It is the directed
    analog of the minimum spanning tree problem.

    References
    ----------
    [1] Khuller, Samir (2002) Advanced Algorithms Lecture 24 Notes.
    https://web.archive.org/web/20121030033722/https://www.cs.umd.edu/class/spring2011/cmsc651/lec07.pdf
    """
    rooted = D.copy()
    rooted.remove_edges_from([(u, root) for u in D.predecessors(root)])
    A = nx.minimum_spanning_arborescence(rooted)
    return A