import pytest
import networkx as nx
class TestDirectedAverageClustering:

    @classmethod
    def setup_class(cls):
        pytest.importorskip('numpy')

    def test_empty(self):
        G = nx.DiGraph()
        with pytest.raises(ZeroDivisionError):
            nx.average_clustering(G)

    def test_average_clustering(self):
        G = nx.cycle_graph(3, create_using=nx.DiGraph())
        G.add_edge(2, 3)
        assert nx.average_clustering(G) == (1 + 1 + 1 / 3) / 8
        assert nx.average_clustering(G, count_zeros=True) == (1 + 1 + 1 / 3) / 8
        assert nx.average_clustering(G, count_zeros=False) == (1 + 1 + 1 / 3) / 6
        assert nx.average_clustering(G, [1, 2, 3]) == (1 + 1 / 3) / 6
        assert nx.average_clustering(G, [1, 2, 3], count_zeros=True) == (1 + 1 / 3) / 6
        assert nx.average_clustering(G, [1, 2, 3], count_zeros=False) == (1 + 1 / 3) / 4