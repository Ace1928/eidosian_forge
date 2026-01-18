import pytest
import networkx as nx
class TestClustering:

    @classmethod
    def setup_class(cls):
        pytest.importorskip('numpy')

    def test_clustering(self):
        G = nx.Graph()
        assert list(nx.clustering(G).values()) == []
        assert nx.clustering(G) == {}

    def test_path(self):
        G = nx.path_graph(10)
        assert list(nx.clustering(G).values()) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert nx.clustering(G) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    def test_cubical(self):
        G = nx.cubical_graph()
        assert list(nx.clustering(G).values()) == [0, 0, 0, 0, 0, 0, 0, 0]
        assert nx.clustering(G, 1) == 0
        assert list(nx.clustering(G, [1, 2]).values()) == [0, 0]
        assert nx.clustering(G, 1) == 0
        assert nx.clustering(G, [1, 2]) == {1: 0, 2: 0}

    def test_k5(self):
        G = nx.complete_graph(5)
        assert list(nx.clustering(G).values()) == [1, 1, 1, 1, 1]
        assert nx.average_clustering(G) == 1
        G.remove_edge(1, 2)
        assert list(nx.clustering(G).values()) == [5 / 6, 1, 1, 5 / 6, 5 / 6]
        assert nx.clustering(G, [1, 4]) == {1: 1, 4: 0.8333333333333334}

    def test_k5_signed(self):
        G = nx.complete_graph(5)
        assert list(nx.clustering(G).values()) == [1, 1, 1, 1, 1]
        assert nx.average_clustering(G) == 1
        G.remove_edge(1, 2)
        G.add_edge(0, 1, weight=-1)
        assert list(nx.clustering(G, weight='weight').values()) == [1 / 6, -1 / 3, 1, 3 / 6, 3 / 6]