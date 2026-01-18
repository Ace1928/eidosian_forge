from networkx import DiGraph, complete_graph, empty_graph, path_graph, wiener_index
class TestWienerIndex:
    """Unit tests for computing the Wiener index of a graph."""

    def test_disconnected_graph(self):
        """Tests that the Wiener index of a disconnected graph is
        positive infinity.

        """
        assert wiener_index(empty_graph(2)) == float('inf')

    def test_directed(self):
        """Tests that each pair of nodes in the directed graph is
        counted once when computing the Wiener index.

        """
        G = complete_graph(3)
        H = DiGraph(G)
        assert 2 * wiener_index(G) == wiener_index(H)

    def test_complete_graph(self):
        """Tests that the Wiener index of the complete graph is simply
        the number of edges.

        """
        n = 10
        G = complete_graph(n)
        assert wiener_index(G) == n * (n - 1) / 2

    def test_path_graph(self):
        """Tests that the Wiener index of the path graph is correctly
        computed.

        """
        n = 9
        G = path_graph(n)
        expected = 2 * sum((i * (n - i) for i in range(1, n // 2 + 1)))
        actual = wiener_index(G)
        assert expected == actual