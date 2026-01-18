from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function
from networkx.utils import py_random_state
from networkx.utils.decorators import not_implemented_for
Adds the corrected betweenness centrality (BC) values for multigraphs.

    Parameters
    ----------
    G : NetworkX graph.

    betweenness : dictionary
        Dictionary mapping adjacent node tuples to betweenness centrality values.

    weight : string or function
        See `_weight_function` for details. Defaults to `None`.

    Returns
    -------
    edges : dictionary
        The parameter `betweenness` including edges with keys and their
        betweenness centrality values.

    The BC value is divided among edges of equal weight.
    