import math
import random
from itertools import combinations
import pytest
import networkx as nx
class TestSoftRandomGeometricGraph:
    """Unit tests for :func:`~networkx.soft_random_geometric_graph`"""

    def test_number_of_nodes(self):
        G = nx.soft_random_geometric_graph(50, 0.25, seed=42)
        assert len(G) == 50
        G = nx.soft_random_geometric_graph(range(50), 0.25, seed=42)
        assert len(G) == 50

    def test_distances(self):
        """Tests that pairs of vertices adjacent if and only if they are
        within the prescribed radius.
        """
        G = nx.soft_random_geometric_graph(50, 0.25)
        for u, v in combinations(G, 2):
            if v in G[u]:
                assert math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25

    def test_p(self):
        """Tests for providing an alternate distance metric to the generator."""

        def dist(x, y):
            return sum((abs(a - b) for a, b in zip(x, y)))
        G = nx.soft_random_geometric_graph(50, 0.25, p=1)
        for u, v in combinations(G, 2):
            if v in G[u]:
                assert dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25

    def test_node_names(self):
        """Tests using values other than sequential numbers as node IDs."""
        import string
        nodes = list(string.ascii_lowercase)
        G = nx.soft_random_geometric_graph(nodes, 0.25)
        assert len(G) == len(nodes)
        for u, v in combinations(G, 2):
            if v in G[u]:
                assert math.dist(G.nodes[u]['pos'], G.nodes[v]['pos']) <= 0.25

    def test_p_dist_default(self):
        """Tests default p_dict = 0.5 returns graph with edge count <= RGG with
        same n, radius, dim and positions
        """
        nodes = 50
        dim = 2
        pos = {v: [random.random() for i in range(dim)] for v in range(nodes)}
        RGG = nx.random_geometric_graph(50, 0.25, pos=pos)
        SRGG = nx.soft_random_geometric_graph(50, 0.25, pos=pos)
        assert len(SRGG.edges()) <= len(RGG.edges())

    def test_p_dist_zero(self):
        """Tests if p_dict = 0 returns disconnected graph with 0 edges"""

        def p_dist(dist):
            return 0
        G = nx.soft_random_geometric_graph(50, 0.25, p_dist=p_dist)
        assert len(G.edges) == 0

    def test_pos_name(self):
        G = nx.soft_random_geometric_graph(50, 0.25, seed=42, pos_name='coords')
        assert all((len(d['coords']) == 2 for n, d in G.nodes.items()))