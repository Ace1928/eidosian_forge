import pytest
import networkx as nx
class TestAverageShortestPathLengthNumpy:

    @classmethod
    def setup_class(cls):
        global np
        import pytest
        np = pytest.importorskip('numpy')

    def test_specified_methods_numpy(self):
        G = nx.Graph()
        nx.add_cycle(G, range(7), weight=2)
        ans = nx.average_shortest_path_length(G, weight='weight', method='floyd-warshall-numpy')
        np.testing.assert_almost_equal(ans, 4)
        G = nx.Graph()
        nx.add_path(G, range(5), weight=2)
        ans = nx.average_shortest_path_length(G, weight='weight', method='floyd-warshall-numpy')
        np.testing.assert_almost_equal(ans, 4)