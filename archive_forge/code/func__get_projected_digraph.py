from pyomo.common.dependencies import networkx_available
def _get_projected_digraph(bg, matching, top_nodes):
    digraph = DiGraph()
    digraph.add_nodes_from(top_nodes)
    for n in top_nodes:
        if n in matching:
            for t in bg[matching[n]]:
                if t != n:
                    digraph.add_edge(t, n)
        for b in bg[n]:
            if b in matching and matching[b] != n:
                digraph.add_edge(n, matching[b])
    return digraph