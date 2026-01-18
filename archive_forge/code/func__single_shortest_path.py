import warnings
import networkx as nx
def _single_shortest_path(adj, firstlevel, paths, cutoff, join):
    """Returns shortest paths

    Shortest Path helper function
    Parameters
    ----------
        adj : dict
            Adjacency dict or view
        firstlevel : dict
            starting nodes, e.g. {source: 1} or {target: 1}
        paths : dict
            paths for starting nodes, e.g. {source: [source]}
        cutoff : int or float
            level at which we stop the process
        join : function
            function to construct a path from two partial paths. Requires two
            list inputs `p1` and `p2`, and returns a list. Usually returns
            `p1 + p2` (forward from source) or `p2 + p1` (backward from target)
    """
    level = 0
    nextlevel = firstlevel
    while nextlevel and cutoff > level:
        thislevel = nextlevel
        nextlevel = {}
        for v in thislevel:
            for w in adj[v]:
                if w not in paths:
                    paths[w] = join(paths[v], [w])
                    nextlevel[w] = 1
        level += 1
    return paths