import collections
import io
import os
import networkx as nx
from networkx.drawing import nx_pydot
def _common_format(g, edge_notation):
    lines = []
    lines.append('Name: %s' % g.name)
    lines.append('Type: %s' % type(g).__name__)
    lines.append('Frozen: %s' % nx.is_frozen(g))
    lines.append('Density: %0.3f' % nx.density(g))
    lines.append('Nodes: %s' % g.number_of_nodes())
    for n, n_data in g.nodes(data=True):
        if n_data:
            lines.append('  - %s (%s)' % (n, n_data))
        else:
            lines.append('  - %s' % n)
    lines.append('Edges: %s' % g.number_of_edges())
    for u, v, e_data in g.edges(data=True):
        if e_data:
            lines.append('  %s %s %s (%s)' % (u, edge_notation, v, e_data))
        else:
            lines.append('  %s %s %s' % (u, edge_notation, v))
    return lines