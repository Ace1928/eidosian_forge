import pytest
import networkx as nx
from networkx.utils import pairwise
class TestMultiSourceDijkstra:
    """Unit tests for the multi-source dialect of Dijkstra's shortest
    path algorithms.

    """

    def test_no_sources(self):
        with pytest.raises(ValueError):
            nx.multi_source_dijkstra(nx.Graph(), {})

    def test_path_no_sources(self):
        with pytest.raises(ValueError):
            nx.multi_source_dijkstra_path(nx.Graph(), {})

    def test_path_length_no_sources(self):
        with pytest.raises(ValueError):
            nx.multi_source_dijkstra_path_length(nx.Graph(), {})

    @pytest.mark.parametrize('fn', (nx.multi_source_dijkstra_path, nx.multi_source_dijkstra_path_length, nx.multi_source_dijkstra))
    def test_absent_source(self, fn):
        G = nx.path_graph(2)
        with pytest.raises(nx.NodeNotFound):
            fn(G, [3], 0)
        with pytest.raises(nx.NodeNotFound):
            fn(G, [3], 3)

    def test_two_sources(self):
        edges = [(0, 1, 1), (1, 2, 1), (2, 3, 10), (3, 4, 1)]
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        sources = {0, 4}
        distances, paths = nx.multi_source_dijkstra(G, sources)
        expected_distances = {0: 0, 1: 1, 2: 2, 3: 1, 4: 0}
        expected_paths = {0: [0], 1: [0, 1], 2: [0, 1, 2], 3: [4, 3], 4: [4]}
        assert distances == expected_distances
        assert paths == expected_paths

    def test_simple_paths(self):
        G = nx.path_graph(4)
        lengths = nx.multi_source_dijkstra_path_length(G, [0])
        assert lengths == {n: n for n in G}
        paths = nx.multi_source_dijkstra_path(G, [0])
        assert paths == {n: list(range(n + 1)) for n in G}