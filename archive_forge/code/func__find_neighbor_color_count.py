import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@staticmethod
def _find_neighbor_color_count(graph, node, node_color, edge_color):
    """
        For `node` in `graph`, count the number of edges of a specific color
        it has to nodes of a specific color.
        """
    counts = Counter()
    neighbors = graph[node]
    for neighbor in neighbors:
        n_color = node_color[neighbor]
        if (node, neighbor) in edge_color:
            e_color = edge_color[node, neighbor]
        else:
            e_color = edge_color[neighbor, node]
        counts[e_color, n_color] += 1
    return counts