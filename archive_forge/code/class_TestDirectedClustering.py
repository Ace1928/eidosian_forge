import pytest
import networkx as nx
class TestDirectedClustering:

    def test_clustering(self):
        G = nx.DiGraph()
        assert list(nx.clustering(G).values()) == []
        assert nx.clustering(G) == {}

    def test_path(self):
        G = nx.path_graph(10, create_using=nx.DiGraph())
        assert list(nx.clustering(G).values()) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert nx.clustering(G) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        assert nx.clustering(G, 0) == 0

    def test_k5(self):
        G = nx.complete_graph(5, create_using=nx.DiGraph())
        assert list(nx.clustering(G).values()) == [1, 1, 1, 1, 1]
        assert nx.average_clustering(G) == 1
        G.remove_edge(1, 2)
        assert list(nx.clustering(G).values()) == [11 / 12, 1, 1, 11 / 12, 11 / 12]
        assert nx.clustering(G, [1, 4]) == {1: 1, 4: 11 / 12}
        G.remove_edge(2, 1)
        assert list(nx.clustering(G).values()) == [5 / 6, 1, 1, 5 / 6, 5 / 6]
        assert nx.clustering(G, [1, 4]) == {1: 1, 4: 0.8333333333333334}
        assert nx.clustering(G, 4) == 5 / 6

    def test_triangle_and_edge(self):
        G = nx.cycle_graph(3, create_using=nx.DiGraph())
        G.add_edge(0, 4)
        assert nx.clustering(G)[0] == 1 / 6