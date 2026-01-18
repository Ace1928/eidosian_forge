import pytest
import networkx as nx
class TestLocalBridges:
    """Unit tests for the local_bridge function."""

    @classmethod
    def setup_class(cls):
        cls.BB = nx.barbell_graph(4, 0)
        cls.square = nx.cycle_graph(4)
        cls.tri = nx.cycle_graph(3)

    def test_nospan(self):
        expected = {(3, 4), (4, 3)}
        assert next(nx.local_bridges(self.BB, with_span=False)) in expected
        assert set(nx.local_bridges(self.square, with_span=False)) == self.square.edges
        assert list(nx.local_bridges(self.tri, with_span=False)) == []

    def test_no_weight(self):
        inf = float('inf')
        expected = {(3, 4, inf), (4, 3, inf)}
        assert next(nx.local_bridges(self.BB)) in expected
        expected = {(u, v, 3) for u, v in self.square.edges}
        assert set(nx.local_bridges(self.square)) == expected
        assert list(nx.local_bridges(self.tri)) == []

    def test_weight(self):
        inf = float('inf')
        G = self.square.copy()
        G.edges[1, 2]['weight'] = 2
        expected = {(u, v, 5 - wt) for u, v, wt in G.edges(data='weight', default=1)}
        assert set(nx.local_bridges(G, weight='weight')) == expected
        expected = {(u, v, 6) for u, v in G.edges}
        lb = nx.local_bridges(G, weight=lambda u, v, d: 2)
        assert set(lb) == expected