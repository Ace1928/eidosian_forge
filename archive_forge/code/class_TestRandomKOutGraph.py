import pytest
import networkx as nx
from networkx.classes import Graph, MultiDiGraph
from networkx.generators.directed import (
class TestRandomKOutGraph:
    """Unit tests for the
    :func:`~networkx.generators.directed.random_k_out_graph` function.

    """

    def test_regularity(self):
        """Tests that the generated graph is `k`-out-regular."""
        n = 10
        k = 3
        alpha = 1
        G = random_k_out_graph(n, k, alpha)
        assert all((d == k for v, d in G.out_degree()))
        G = random_k_out_graph(n, k, alpha, seed=42)
        assert all((d == k for v, d in G.out_degree()))

    def test_no_self_loops(self):
        """Tests for forbidding self-loops."""
        n = 10
        k = 3
        alpha = 1
        G = random_k_out_graph(n, k, alpha, self_loops=False)
        assert nx.number_of_selfloops(G) == 0

    def test_negative_alpha(self):
        with pytest.raises(ValueError, match='alpha must be positive'):
            random_k_out_graph(10, 3, -1)