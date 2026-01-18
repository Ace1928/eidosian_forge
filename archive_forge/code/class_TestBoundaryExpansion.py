import networkx as nx
class TestBoundaryExpansion:
    """Unit tests for the :func:`~networkx.boundary_expansion` function."""

    def test_graph(self):
        G = nx.complete_graph(10)
        S = set(range(4))
        expansion = nx.boundary_expansion(G, S)
        expected = 6 / 4
        assert expected == expansion