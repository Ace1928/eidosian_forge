import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
class TestGeneratorPrivateFunctions:

    def test_triangles_error(self):
        G = nx.diamond_graph()
        pytest.raises(nx.NetworkXError, line._triangles, G, (4, 0))
        pytest.raises(nx.NetworkXError, line._triangles, G, (0, 3))

    def test_odd_triangles_error(self):
        G = nx.diamond_graph()
        pytest.raises(nx.NetworkXError, line._odd_triangle, G, (0, 1, 4))
        pytest.raises(nx.NetworkXError, line._odd_triangle, G, (0, 1, 3))

    def test_select_starting_cell_error(self):
        G = nx.diamond_graph()
        pytest.raises(nx.NetworkXError, line._select_starting_cell, G, (4, 0))
        pytest.raises(nx.NetworkXError, line._select_starting_cell, G, (0, 3))

    def test_diamond_graph(self):
        G = nx.diamond_graph()
        for edge in G.edges:
            cell = line._select_starting_cell(G, starting_edge=edge)
            assert len(cell) == 3
            assert all((v in G[u] for u in cell for v in cell if u != v))