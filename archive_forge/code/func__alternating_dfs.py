import one of the named maximum matching algorithms directly.
import collections
import itertools
import networkx as nx
from networkx.algorithms.bipartite import sets as bipartite_sets
from networkx.algorithms.bipartite.matrix import biadjacency_matrix
def _alternating_dfs(u, along_matched=True):
    """Returns True if and only if `u` is connected to one of the
        targets by an alternating path.

        `u` is a vertex in the graph `G`.

        If `along_matched` is True, this step of the depth-first search
        will continue only through edges in the given matching. Otherwise, it
        will continue only through edges *not* in the given matching.

        """
    visited = set()
    initial_depth = 0 if along_matched else 1
    stack = [(u, iter(G[u]), initial_depth)]
    while stack:
        parent, children, depth = stack[-1]
        valid_edges = matched_edges if depth % 2 else unmatched_edges
        try:
            child = next(children)
            if child not in visited:
                if (parent, child) in valid_edges or (child, parent) in valid_edges:
                    if child in targets:
                        return True
                    visited.add(child)
                    stack.append((child, iter(G[child]), depth + 1))
        except StopIteration:
            stack.pop()
    return False