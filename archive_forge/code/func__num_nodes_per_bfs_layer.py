import pytest
import networkx as nx
def _num_nodes_per_bfs_layer(pos):
    """Helper function to extract the number of nodes in each layer of bfs_layout"""
    x = np.array(list(pos.values()))[:, 0]
    _, layer_count = np.unique(x, return_counts=True)
    return layer_count