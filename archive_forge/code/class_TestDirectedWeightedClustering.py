import pytest
import networkx as nx
class TestDirectedWeightedClustering:

    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip('numpy')

    def test_clustering(self):
        G = nx.DiGraph()
        assert list(nx.clustering(G, weight='weight').values()) == []
        assert nx.clustering(G) == {}

    def test_path(self):
        G = nx.path_graph(10, create_using=nx.DiGraph())
        assert list(nx.clustering(G, weight='weight').values()) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert nx.clustering(G, weight='weight') == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    def test_k5(self):
        G = nx.complete_graph(5, create_using=nx.DiGraph())
        assert list(nx.clustering(G, weight='weight').values()) == [1, 1, 1, 1, 1]
        assert nx.average_clustering(G, weight='weight') == 1
        G.remove_edge(1, 2)
        assert list(nx.clustering(G, weight='weight').values()) == [11 / 12, 1, 1, 11 / 12, 11 / 12]
        assert nx.clustering(G, [1, 4], weight='weight') == {1: 1, 4: 11 / 12}
        G.remove_edge(2, 1)
        assert list(nx.clustering(G, weight='weight').values()) == [5 / 6, 1, 1, 5 / 6, 5 / 6]
        assert nx.clustering(G, [1, 4], weight='weight') == {1: 1, 4: 0.8333333333333334}

    def test_triangle_and_edge(self):
        G = nx.cycle_graph(3, create_using=nx.DiGraph())
        G.add_edge(0, 4, weight=2)
        assert nx.clustering(G)[0] == 1 / 6
        np.testing.assert_allclose(nx.clustering(G, weight='weight')[0], 1 / 12)
        np.testing.assert_allclose(nx.clustering(G, 0, weight='weight'), 1 / 12)