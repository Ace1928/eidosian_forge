from collections import deque
from heapq import heappop, heappush
from itertools import count
import networkx as nx
from networkx.algorithms.shortest_paths.generic import _build_paths_from_predecessors
def _bellman_ford(G, source, weight, pred=None, paths=None, dist=None, target=None, heuristic=True):
    """Calls relaxation loop for Bellman–Ford algorithm and builds paths

    This is an implementation of the SPFA variant.
    See https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm

    Parameters
    ----------
    G : NetworkX graph

    source: list
        List of source nodes. The shortest path from any of the source
        nodes will be found if multiple sources are provided.

    weight : function
        The weight of an edge is the value returned by the function. The
        function must accept exactly three positional arguments: the two
        endpoints of an edge and the dictionary of edge attributes for
        that edge. The function must return a number.

    pred: dict of lists, optional (default=None)
        dict to store a list of predecessors keyed by that node
        If None, predecessors are not stored

    paths: dict, optional (default=None)
        dict to store the path list from source to each node, keyed by node
        If None, paths are not stored

    dist: dict, optional (default=None)
        dict to store distance from source to the keyed node
        If None, returned dist dict contents default to 0 for every node in the
        source list

    target: node label, optional
        Ending node for path. Path lengths to other destinations may (and
        probably will) be incorrect.

    heuristic : bool
        Determines whether to use a heuristic to early detect negative
        cycles at a hopefully negligible cost.

    Returns
    -------
    dist : dict
        Returns a dict keyed by node to the distance from the source.
        Dicts for paths and pred are in the mutated input dicts by those names.

    Raises
    ------
    NodeNotFound
        If any of `source` is not in `G`.

    NetworkXUnbounded
        If the (di)graph contains a negative (di)cycle, the
        algorithm raises an exception to indicate the presence of the
        negative (di)cycle.  Note: any negative weight edge in an
        undirected graph is a negative cycle
    """
    if pred is None:
        pred = {v: [] for v in source}
    if dist is None:
        dist = {v: 0 for v in source}
    negative_cycle_found = _inner_bellman_ford(G, source, weight, pred, dist, heuristic)
    if negative_cycle_found is not None:
        raise nx.NetworkXUnbounded('Negative cycle detected.')
    if paths is not None:
        sources = set(source)
        dsts = [target] if target is not None else pred
        for dst in dsts:
            gen = _build_paths_from_predecessors(sources, dst, pred)
            paths[dst] = next(gen)
    return dist