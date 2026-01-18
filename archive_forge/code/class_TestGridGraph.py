from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal
class TestGridGraph:
    """Unit tests for :func:`networkx.generators.lattice.grid_graph`"""

    def test_grid_graph(self):
        """grid_graph([n,m]) is a connected simple graph with the
        following properties:
        number_of_nodes = n*m
        degree_histogram = [0,0,4,2*(n+m)-8,(n-2)*(m-2)]
        """
        for n, m in [(3, 5), (5, 3), (4, 5), (5, 4)]:
            dim = [n, m]
            g = nx.grid_graph(dim)
            assert len(g) == n * m
            assert nx.degree_histogram(g) == [0, 0, 4, 2 * (n + m) - 8, (n - 2) * (m - 2)]
        for n, m in [(1, 5), (5, 1)]:
            dim = [n, m]
            g = nx.grid_graph(dim)
            assert len(g) == n * m
            assert nx.is_isomorphic(g, nx.path_graph(5))

    def test_node_input(self):
        G = nx.grid_graph([range(7, 9), range(3, 6)])
        assert len(G) == 2 * 3
        assert nx.is_isomorphic(G, nx.grid_graph([2, 3]))

    def test_periodic_iterable(self):
        m, n, k = (3, 7, 5)
        for a, b, c in product([0, 1], [0, 1], [0, 1]):
            G = nx.grid_graph([m, n, k], periodic=(a, b, c))
            num_e = (m + a - 1) * n * k + (n + b - 1) * m * k + (k + c - 1) * m * n
            assert G.number_of_nodes() == m * n * k
            assert G.number_of_edges() == num_e