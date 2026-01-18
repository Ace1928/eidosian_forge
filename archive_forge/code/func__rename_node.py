import numpy as np
import heapq
def _rename_node(graph, node_id, copy_id):
    """Rename `node_id` in `graph` to `copy_id`."""
    graph._add_node_silent(copy_id)
    graph.nodes[copy_id].update(graph.nodes[node_id])
    for nbr in graph.neighbors(node_id):
        wt = graph[node_id][nbr]['weight']
        graph.add_edge(nbr, copy_id, {'weight': wt})
    graph.remove_node(node_id)