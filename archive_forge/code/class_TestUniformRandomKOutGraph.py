import pytest
import networkx as nx
from networkx.classes import Graph, MultiDiGraph
from networkx.generators.directed import (
class TestUniformRandomKOutGraph:
    """Unit tests for the
    :func:`~networkx.generators.directed.random_uniform_k_out_graph`
    function.

    """

    def test_regularity(self):
        """Tests that the generated graph is `k`-out-regular."""
        n = 10
        k = 3
        G = random_uniform_k_out_graph(n, k)
        assert all((d == k for v, d in G.out_degree()))
        G = random_uniform_k_out_graph(n, k, seed=42)
        assert all((d == k for v, d in G.out_degree()))

    def test_no_self_loops(self):
        """Tests for forbidding self-loops."""
        n = 10
        k = 3
        G = random_uniform_k_out_graph(n, k, self_loops=False)
        assert nx.number_of_selfloops(G) == 0
        assert all((d == k for v, d in G.out_degree()))

    def test_with_replacement(self):
        n = 10
        k = 3
        G = random_uniform_k_out_graph(n, k, with_replacement=True)
        assert G.is_multigraph()
        assert all((d == k for v, d in G.out_degree()))
        n = 10
        k = 9
        G = random_uniform_k_out_graph(n, k, with_replacement=False, self_loops=False)
        assert nx.number_of_selfloops(G) == 0
        assert all((d == k for v, d in G.out_degree()))

    def test_without_replacement(self):
        n = 10
        k = 3
        G = random_uniform_k_out_graph(n, k, with_replacement=False)
        assert not G.is_multigraph()
        assert all((d == k for v, d in G.out_degree()))