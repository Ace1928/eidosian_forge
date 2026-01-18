import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestPrim(MultigraphMSTTestBase):
    """Unit tests for computing a minimum (or maximum) spanning tree
    using Prim's algorithm.
    """
    algorithm = 'prim'

    def test_prim_mst_edges_simple_graph(self):
        H = nx.Graph()
        H.add_edge(1, 2, key=2, weight=3)
        H.add_edge(3, 2, key=1, weight=2)
        H.add_edge(3, 1, key=1, weight=4)
        mst_edges = nx.minimum_spanning_edges(H, algorithm=self.algo, ignore_nan=True)
        assert edges_equal([(1, 2, {'key': 2, 'weight': 3}), (2, 3, {'key': 1, 'weight': 2})], list(mst_edges))

    def test_ignore_nan(self):
        """Tests that the edges with NaN weights are ignored or
        raise an Error based on ignore_nan is true or false"""
        H = nx.MultiGraph()
        H.add_edge(1, 2, key=1, weight=float('nan'))
        H.add_edge(1, 2, key=2, weight=3)
        H.add_edge(3, 2, key=1, weight=2)
        H.add_edge(3, 1, key=1, weight=4)
        mst_edges = nx.minimum_spanning_edges(H, algorithm=self.algo, ignore_nan=True)
        assert edges_equal([(1, 2, 2, {'weight': 3}), (2, 3, 1, {'weight': 2})], list(mst_edges))
        with pytest.raises(ValueError):
            list(nx.minimum_spanning_edges(H, algorithm=self.algo, ignore_nan=False))

    def test_multigraph_keys_tree(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, key='a', weight=2)
        G.add_edge(0, 1, key='b', weight=1)
        T = nx.minimum_spanning_tree(G, algorithm=self.algo)
        assert edges_equal([(0, 1, 1)], list(T.edges(data='weight')))

    def test_multigraph_keys_tree_max(self):
        G = nx.MultiGraph()
        G.add_edge(0, 1, key='a', weight=2)
        G.add_edge(0, 1, key='b', weight=1)
        T = nx.maximum_spanning_tree(G, algorithm=self.algo)
        assert edges_equal([(0, 1, 2)], list(T.edges(data='weight')))