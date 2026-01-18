import pytest
import networkx as nx
class TestWeightedClustering:

    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip('numpy')

    def test_clustering(self):
        G = nx.Graph()
        assert list(nx.clustering(G, weight='weight').values()) == []
        assert nx.clustering(G) == {}

    def test_path(self):
        G = nx.path_graph(10)
        assert list(nx.clustering(G, weight='weight').values()) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert nx.clustering(G, weight='weight') == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    def test_cubical(self):
        G = nx.cubical_graph()
        assert list(nx.clustering(G, weight='weight').values()) == [0, 0, 0, 0, 0, 0, 0, 0]
        assert nx.clustering(G, 1) == 0
        assert list(nx.clustering(G, [1, 2], weight='weight').values()) == [0, 0]
        assert nx.clustering(G, 1, weight='weight') == 0
        assert nx.clustering(G, [1, 2], weight='weight') == {1: 0, 2: 0}

    def test_k5(self):
        G = nx.complete_graph(5)
        assert list(nx.clustering(G, weight='weight').values()) == [1, 1, 1, 1, 1]
        assert nx.average_clustering(G, weight='weight') == 1
        G.remove_edge(1, 2)
        assert list(nx.clustering(G, weight='weight').values()) == [5 / 6, 1, 1, 5 / 6, 5 / 6]
        assert nx.clustering(G, [1, 4], weight='weight') == {1: 1, 4: 0.8333333333333334}

    def test_triangle_and_edge(self):
        G = nx.cycle_graph(3)
        G.add_edge(0, 4, weight=2)
        assert nx.clustering(G)[0] == 1 / 3
        np.testing.assert_allclose(nx.clustering(G, weight='weight')[0], 1 / 6)
        np.testing.assert_allclose(nx.clustering(G, 0, weight='weight'), 1 / 6)

    def test_triangle_and_signed_edge(self):
        G = nx.cycle_graph(3)
        G.add_edge(0, 1, weight=-1)
        G.add_edge(3, 0, weight=0)
        assert nx.clustering(G)[0] == 1 / 3
        assert nx.clustering(G, weight='weight')[0] == -1 / 3