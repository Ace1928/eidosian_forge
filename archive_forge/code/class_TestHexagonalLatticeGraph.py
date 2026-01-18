from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal
class TestHexagonalLatticeGraph:
    """Tests for :func:`networkx.generators.lattice.hexagonal_lattice_graph`"""

    def test_lattice_points(self):
        """Tests that the graph is really a hexagonal lattice."""
        for m, n in [(4, 5), (4, 4), (4, 3), (3, 2), (3, 3), (3, 5)]:
            G = nx.hexagonal_lattice_graph(m, n)
            assert len(G) == 2 * (m + 1) * (n + 1) - 2
        C_6 = nx.cycle_graph(6)
        hexagons = [[(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)], [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4)], [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)], [(2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)], [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4)]]
        for hexagon in hexagons:
            assert nx.is_isomorphic(G.subgraph(hexagon), C_6)

    def test_directed(self):
        """Tests for creating a directed hexagonal lattice."""
        G = nx.hexagonal_lattice_graph(3, 5, create_using=nx.Graph())
        H = nx.hexagonal_lattice_graph(3, 5, create_using=nx.DiGraph())
        assert H.is_directed()
        pos = nx.get_node_attributes(H, 'pos')
        for u, v in H.edges():
            assert pos[v][1] >= pos[u][1]
            if pos[v][1] == pos[u][1]:
                assert pos[v][0] > pos[u][0]

    def test_multigraph(self):
        """Tests for creating a hexagonal lattice multigraph."""
        G = nx.hexagonal_lattice_graph(3, 5, create_using=nx.Graph())
        H = nx.hexagonal_lattice_graph(3, 5, create_using=nx.MultiGraph())
        assert list(H.edges()) == list(G.edges())

    def test_periodic(self):
        G = nx.hexagonal_lattice_graph(4, 6, periodic=True)
        assert len(G) == 48
        assert G.size() == 72
        assert len([n for n, d in G.degree() if d != 3]) == 0
        G = nx.hexagonal_lattice_graph(5, 8, periodic=True)
        HLG = nx.hexagonal_lattice_graph
        pytest.raises(nx.NetworkXError, HLG, 2, 7, periodic=True)
        pytest.raises(nx.NetworkXError, HLG, 1, 4, periodic=True)
        pytest.raises(nx.NetworkXError, HLG, 2, 1, periodic=True)