import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class MinimumSpanningTreeTestBase:
    """Base class for test classes for minimum spanning tree algorithms.
    This class contains some common tests that will be inherited by
    subclasses. Each subclass must have a class attribute
    :data:`algorithm` that is a string representing the algorithm to
    run, as described under the ``algorithm`` keyword argument for the
    :func:`networkx.minimum_spanning_edges` function.  Subclasses can
    then implement any algorithm-specific tests.
    """

    def setup_method(self, method):
        """Creates an example graph and stores the expected minimum and
        maximum spanning tree edges.
        """
        self.algo = self.algorithm
        edges = [(0, 1, 7), (0, 3, 5), (1, 2, 8), (1, 3, 9), (1, 4, 7), (2, 4, 5), (3, 4, 15), (3, 5, 6), (4, 5, 8), (4, 6, 9), (5, 6, 11)]
        self.G = nx.Graph()
        self.G.add_weighted_edges_from(edges)
        self.minimum_spanning_edgelist = [(0, 1, {'weight': 7}), (0, 3, {'weight': 5}), (1, 4, {'weight': 7}), (2, 4, {'weight': 5}), (3, 5, {'weight': 6}), (4, 6, {'weight': 9})]
        self.maximum_spanning_edgelist = [(0, 1, {'weight': 7}), (1, 2, {'weight': 8}), (1, 3, {'weight': 9}), (3, 4, {'weight': 15}), (4, 6, {'weight': 9}), (5, 6, {'weight': 11})]

    def test_minimum_edges(self):
        edges = nx.minimum_spanning_edges(self.G, algorithm=self.algo)
        actual = sorted(((min(u, v), max(u, v), d) for u, v, d in edges))
        assert edges_equal(actual, self.minimum_spanning_edgelist)

    def test_maximum_edges(self):
        edges = nx.maximum_spanning_edges(self.G, algorithm=self.algo)
        actual = sorted(((min(u, v), max(u, v), d) for u, v, d in edges))
        assert edges_equal(actual, self.maximum_spanning_edgelist)

    def test_without_data(self):
        edges = nx.minimum_spanning_edges(self.G, algorithm=self.algo, data=False)
        actual = sorted(((min(u, v), max(u, v)) for u, v in edges))
        expected = [(u, v) for u, v, d in self.minimum_spanning_edgelist]
        assert edges_equal(actual, expected)

    def test_nan_weights(self):
        G = self.G
        G.add_edge(0, 12, weight=float('nan'))
        edges = nx.minimum_spanning_edges(G, algorithm=self.algo, data=False, ignore_nan=True)
        actual = sorted(((min(u, v), max(u, v)) for u, v in edges))
        expected = [(u, v) for u, v, d in self.minimum_spanning_edgelist]
        assert edges_equal(actual, expected)
        edges = nx.minimum_spanning_edges(G, algorithm=self.algo, data=False, ignore_nan=False)
        with pytest.raises(ValueError):
            list(edges)
        edges = nx.minimum_spanning_edges(G, algorithm=self.algo, data=False)
        with pytest.raises(ValueError):
            list(edges)

    def test_nan_weights_MultiGraph(self):
        G = nx.MultiGraph()
        G.add_edge(0, 12, weight=float('nan'))
        edges = nx.minimum_spanning_edges(G, algorithm='prim', data=False, ignore_nan=False)
        with pytest.raises(ValueError):
            list(edges)
        edges = nx.minimum_spanning_edges(G, algorithm='prim', data=False)
        with pytest.raises(ValueError):
            list(edges)

    def test_nan_weights_order(self):
        edges = [(0, 1, 7), (0, 3, 5), (1, 2, 8), (1, 3, 9), (1, 4, 7), (2, 4, 5), (3, 4, 15), (3, 5, 6), (4, 5, 8), (4, 6, 9), (5, 6, 11)]
        G = nx.Graph()
        G.add_weighted_edges_from([(u + 1, v + 1, wt) for u, v, wt in edges])
        G.add_edge(0, 7, weight=float('nan'))
        edges = nx.minimum_spanning_edges(G, algorithm=self.algo, data=False, ignore_nan=True)
        actual = sorted(((min(u, v), max(u, v)) for u, v in edges))
        shift = [(u + 1, v + 1) for u, v, d in self.minimum_spanning_edgelist]
        assert edges_equal(actual, shift)

    def test_isolated_node(self):
        edges = [(0, 1, 7), (0, 3, 5), (1, 2, 8), (1, 3, 9), (1, 4, 7), (2, 4, 5), (3, 4, 15), (3, 5, 6), (4, 5, 8), (4, 6, 9), (5, 6, 11)]
        G = nx.Graph()
        G.add_weighted_edges_from([(u + 1, v + 1, wt) for u, v, wt in edges])
        G.add_node(0)
        edges = nx.minimum_spanning_edges(G, algorithm=self.algo, data=False, ignore_nan=True)
        actual = sorted(((min(u, v), max(u, v)) for u, v in edges))
        shift = [(u + 1, v + 1) for u, v, d in self.minimum_spanning_edgelist]
        assert edges_equal(actual, shift)

    def test_minimum_tree(self):
        T = nx.minimum_spanning_tree(self.G, algorithm=self.algo)
        actual = sorted(T.edges(data=True))
        assert edges_equal(actual, self.minimum_spanning_edgelist)

    def test_maximum_tree(self):
        T = nx.maximum_spanning_tree(self.G, algorithm=self.algo)
        actual = sorted(T.edges(data=True))
        assert edges_equal(actual, self.maximum_spanning_edgelist)

    def test_disconnected(self):
        G = nx.Graph([(0, 1, {'weight': 1}), (2, 3, {'weight': 2})])
        T = nx.minimum_spanning_tree(G, algorithm=self.algo)
        assert nodes_equal(list(T), list(range(4)))
        assert edges_equal(list(T.edges()), [(0, 1), (2, 3)])

    def test_empty_graph(self):
        G = nx.empty_graph(3)
        T = nx.minimum_spanning_tree(G, algorithm=self.algo)
        assert nodes_equal(sorted(T), list(range(3)))
        assert T.number_of_edges() == 0

    def test_attributes(self):
        G = nx.Graph()
        G.add_edge(1, 2, weight=1, color='red', distance=7)
        G.add_edge(2, 3, weight=1, color='green', distance=2)
        G.add_edge(1, 3, weight=10, color='blue', distance=1)
        G.graph['foo'] = 'bar'
        T = nx.minimum_spanning_tree(G, algorithm=self.algo)
        assert T.graph == G.graph
        assert nodes_equal(T, G)
        for u, v in T.edges():
            assert T.adj[u][v] == G.adj[u][v]

    def test_weight_attribute(self):
        G = nx.Graph()
        G.add_edge(0, 1, weight=1, distance=7)
        G.add_edge(0, 2, weight=30, distance=1)
        G.add_edge(1, 2, weight=1, distance=1)
        G.add_node(3)
        T = nx.minimum_spanning_tree(G, algorithm=self.algo, weight='distance')
        assert nodes_equal(sorted(T), list(range(4)))
        assert edges_equal(sorted(T.edges()), [(0, 2), (1, 2)])
        T = nx.maximum_spanning_tree(G, algorithm=self.algo, weight='distance')
        assert nodes_equal(sorted(T), list(range(4)))
        assert edges_equal(sorted(T.edges()), [(0, 1), (0, 2)])