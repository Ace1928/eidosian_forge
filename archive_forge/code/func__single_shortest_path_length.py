import warnings
import networkx as nx
def _single_shortest_path_length(adj, firstlevel, cutoff):
    """Yields (node, level) in a breadth first search

    Shortest Path Length helper function
    Parameters
    ----------
        adj : dict
            Adjacency dict or view
        firstlevel : list
            starting nodes, e.g. [source] or [target]
        cutoff : int or float
            level at which we stop the process
    """
    seen = set(firstlevel)
    nextlevel = firstlevel
    level = 0
    n = len(adj)
    for v in nextlevel:
        yield (v, level)
    while nextlevel and cutoff > level:
        level += 1
        thislevel = nextlevel
        nextlevel = []
        for v in thislevel:
            for w in adj[v]:
                if w not in seen:
                    seen.add(w)
                    nextlevel.append(w)
                    yield (w, level)
            if len(seen) == n:
                return