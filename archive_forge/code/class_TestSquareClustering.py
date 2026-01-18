import pytest
import networkx as nx
class TestSquareClustering:

    def test_clustering(self):
        G = nx.Graph()
        assert list(nx.square_clustering(G).values()) == []
        assert nx.square_clustering(G) == {}

    def test_path(self):
        G = nx.path_graph(10)
        assert list(nx.square_clustering(G).values()) == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert nx.square_clustering(G) == {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}

    def test_cubical(self):
        G = nx.cubical_graph()
        assert list(nx.square_clustering(G).values()) == [1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3, 1 / 3]
        assert list(nx.square_clustering(G, [1, 2]).values()) == [1 / 3, 1 / 3]
        assert nx.square_clustering(G, [1])[1] == 1 / 3
        assert nx.square_clustering(G, 1) == 1 / 3
        assert nx.square_clustering(G, [1, 2]) == {1: 1 / 3, 2: 1 / 3}

    def test_k5(self):
        G = nx.complete_graph(5)
        assert list(nx.square_clustering(G).values()) == [1, 1, 1, 1, 1]

    def test_bipartite_k5(self):
        G = nx.complete_bipartite_graph(5, 5)
        assert list(nx.square_clustering(G).values()) == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    def test_lind_square_clustering(self):
        """Test C4 for figure 1 Lind et al (2005)"""
        G = nx.Graph([(1, 2), (1, 3), (1, 6), (1, 7), (2, 4), (2, 5), (3, 4), (3, 5), (6, 7), (7, 8), (6, 8), (7, 9), (7, 10), (6, 11), (6, 12), (2, 13), (2, 14), (3, 15), (3, 16)])
        G1 = G.subgraph([1, 2, 3, 4, 5, 13, 14, 15, 16])
        G2 = G.subgraph([1, 6, 7, 8, 9, 10, 11, 12])
        assert nx.square_clustering(G, [1])[1] == 3 / 43
        assert nx.square_clustering(G1, [1])[1] == 2 / 6
        assert nx.square_clustering(G2, [1])[1] == 1 / 5

    def test_peng_square_clustering(self):
        """Test eq2 for figure 1 Peng et al (2008)"""
        G = nx.Graph([(1, 2), (1, 3), (2, 4), (3, 4), (3, 5), (3, 6)])
        assert nx.square_clustering(G, [1])[1] == 1 / 3