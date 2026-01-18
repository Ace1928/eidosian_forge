from collections import defaultdict, deque
import networkx as nx
from networkx.algorithms.community import modularity
from networkx.utils import py_random_state
def _gen_graph(G, partition):
    """Generate a new graph based on the partitions of a given graph"""
    H = G.__class__()
    node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        for node in part:
            node2com[node] = i
            nodes.update(G.nodes[node].get('nodes', {node}))
        H.add_node(i, nodes=nodes)
    for node1, node2, wt in G.edges(data=True):
        wt = wt['weight']
        com1 = node2com[node1]
        com2 = node2com[node2]
        temp = H.get_edge_data(com1, com2, {'weight': 0})['weight']
        H.add_edge(com1, com2, weight=wt + temp)
    return H