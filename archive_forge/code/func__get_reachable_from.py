from pyomo.common.dependencies import networkx_available
def _get_reachable_from(digraph, sources):
    _filter = set()
    reachable = []
    for node in sources:
        for i, j in bfs_edges(digraph, node):
            if j not in _filter:
                _filter.add(j)
                reachable.append(j)
    return (reachable, _filter)