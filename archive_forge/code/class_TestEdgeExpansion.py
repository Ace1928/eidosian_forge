import networkx as nx
class TestEdgeExpansion:
    """Unit tests for the :func:`~networkx.edge_expansion` function."""

    def test_graph(self):
        G = nx.barbell_graph(5, 0)
        S = set(range(5))
        T = set(G) - S
        expansion = nx.edge_expansion(G, S, T)
        expected = 1 / 5
        assert expected == expansion
        assert expected == nx.edge_expansion(G, S)