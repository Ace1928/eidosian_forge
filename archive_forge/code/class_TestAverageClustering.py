import pytest
import networkx as nx
class TestAverageClustering:

    @classmethod
    def setup_class(cls):
        pytest.importorskip('numpy')

    def test_empty(self):
        G = nx.Graph()
        with pytest.raises(ZeroDivisionError):
            nx.average_clustering(G)

    def test_average_clustering(self):
        G = nx.cycle_graph(3)
        G.add_edge(2, 3)
        assert nx.average_clustering(G) == (1 + 1 + 1 / 3) / 4
        assert nx.average_clustering(G, count_zeros=True) == (1 + 1 + 1 / 3) / 4
        assert nx.average_clustering(G, count_zeros=False) == (1 + 1 + 1 / 3) / 3
        assert nx.average_clustering(G, [1, 2, 3]) == (1 + 1 / 3) / 3
        assert nx.average_clustering(G, [1, 2, 3], count_zeros=True) == (1 + 1 / 3) / 3
        assert nx.average_clustering(G, [1, 2, 3], count_zeros=False) == (1 + 1 / 3) / 2

    def test_average_clustering_signed(self):
        G = nx.cycle_graph(3)
        G.add_edge(2, 3)
        G.add_edge(0, 1, weight=-1)
        assert nx.average_clustering(G, weight='weight') == (-1 - 1 - 1 / 3) / 4
        assert nx.average_clustering(G, weight='weight', count_zeros=True) == (-1 - 1 - 1 / 3) / 4
        assert nx.average_clustering(G, weight='weight', count_zeros=False) == (-1 - 1 - 1 / 3) / 3