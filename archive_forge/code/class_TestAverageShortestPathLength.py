import pytest
import networkx as nx
class TestAverageShortestPathLength:

    def test_cycle_graph(self):
        ans = nx.average_shortest_path_length(nx.cycle_graph(7))
        assert ans == pytest.approx(2, abs=1e-07)

    def test_path_graph(self):
        ans = nx.average_shortest_path_length(nx.path_graph(5))
        assert ans == pytest.approx(2, abs=1e-07)

    def test_weighted(self):
        G = nx.Graph()
        nx.add_cycle(G, range(7), weight=2)
        ans = nx.average_shortest_path_length(G, weight='weight')
        assert ans == pytest.approx(4, abs=1e-07)
        G = nx.Graph()
        nx.add_path(G, range(5), weight=2)
        ans = nx.average_shortest_path_length(G, weight='weight')
        assert ans == pytest.approx(4, abs=1e-07)

    def test_specified_methods(self):
        G = nx.Graph()
        nx.add_cycle(G, range(7), weight=2)
        ans = nx.average_shortest_path_length(G, weight='weight', method='dijkstra')
        assert ans == pytest.approx(4, abs=1e-07)
        ans = nx.average_shortest_path_length(G, weight='weight', method='bellman-ford')
        assert ans == pytest.approx(4, abs=1e-07)
        ans = nx.average_shortest_path_length(G, weight='weight', method='floyd-warshall')
        assert ans == pytest.approx(4, abs=1e-07)
        G = nx.Graph()
        nx.add_path(G, range(5), weight=2)
        ans = nx.average_shortest_path_length(G, weight='weight', method='dijkstra')
        assert ans == pytest.approx(4, abs=1e-07)
        ans = nx.average_shortest_path_length(G, weight='weight', method='bellman-ford')
        assert ans == pytest.approx(4, abs=1e-07)
        ans = nx.average_shortest_path_length(G, weight='weight', method='floyd-warshall')
        assert ans == pytest.approx(4, abs=1e-07)

    def test_directed_not_strongly_connected(self):
        G = nx.DiGraph([(0, 1)])
        with pytest.raises(nx.NetworkXError, match='Graph is not strongly connected'):
            nx.average_shortest_path_length(G)

    def test_undirected_not_connected(self):
        g = nx.Graph()
        g.add_nodes_from(range(3))
        g.add_edge(0, 1)
        pytest.raises(nx.NetworkXError, nx.average_shortest_path_length, g)

    def test_trivial_graph(self):
        """Tests that the trivial graph has average path length zero,
        since there is exactly one path of length zero in the trivial
        graph.

        For more information, see issue #1960.

        """
        G = nx.trivial_graph()
        assert nx.average_shortest_path_length(G) == 0

    def test_null_graph(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.average_shortest_path_length(nx.null_graph())

    def test_bad_method(self):
        with pytest.raises(ValueError):
            G = nx.path_graph(2)
            nx.average_shortest_path_length(G, weight='weight', method='SPAM')