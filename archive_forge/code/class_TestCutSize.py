import networkx as nx
class TestCutSize:
    """Unit tests for the :func:`~networkx.cut_size` function."""

    def test_symmetric(self):
        """Tests that the cut size is symmetric."""
        G = nx.barbell_graph(3, 0)
        S = {0, 1, 4}
        T = {2, 3, 5}
        assert nx.cut_size(G, S, T) == 4
        assert nx.cut_size(G, T, S) == 4

    def test_single_edge(self):
        """Tests for a cut of a single edge."""
        G = nx.barbell_graph(3, 0)
        S = {0, 1, 2}
        T = {3, 4, 5}
        assert nx.cut_size(G, S, T) == 1
        assert nx.cut_size(G, T, S) == 1

    def test_directed(self):
        """Tests that each directed edge is counted once in the cut."""
        G = nx.barbell_graph(3, 0).to_directed()
        S = {0, 1, 2}
        T = {3, 4, 5}
        assert nx.cut_size(G, S, T) == 2
        assert nx.cut_size(G, T, S) == 2

    def test_directed_symmetric(self):
        """Tests that a cut in a directed graph is symmetric."""
        G = nx.barbell_graph(3, 0).to_directed()
        S = {0, 1, 4}
        T = {2, 3, 5}
        assert nx.cut_size(G, S, T) == 8
        assert nx.cut_size(G, T, S) == 8

    def test_multigraph(self):
        """Tests that parallel edges are each counted for a cut."""
        G = nx.MultiGraph(['ab', 'ab'])
        assert nx.cut_size(G, {'a'}, {'b'}) == 2