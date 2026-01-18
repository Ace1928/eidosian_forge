from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal
class TestTriangularLatticeGraph:
    """Tests for :func:`networkx.generators.lattice.triangular_lattice_graph`"""

    def test_lattice_points(self):
        """Tests that the graph is really a triangular lattice."""
        for m, n in [(2, 3), (2, 2), (2, 1), (3, 3), (3, 2), (3, 4)]:
            G = nx.triangular_lattice_graph(m, n)
            N = (n + 1) // 2
            assert len(G) == (m + 1) * (1 + N) - n % 2 * ((m + 1) // 2)
        for i, j in G.nodes():
            nbrs = G[i, j]
            if i < N:
                assert (i + 1, j) in nbrs
            if j < m:
                assert (i, j + 1) in nbrs
            if j < m and (i > 0 or j % 2) and (i < N or (j + 1) % 2):
                assert (i + 1, j + 1) in nbrs or (i - 1, j + 1) in nbrs

    def test_directed(self):
        """Tests for creating a directed triangular lattice."""
        G = nx.triangular_lattice_graph(3, 4, create_using=nx.Graph())
        H = nx.triangular_lattice_graph(3, 4, create_using=nx.DiGraph())
        assert H.is_directed()
        for u, v in H.edges():
            assert v[1] >= u[1]
            if v[1] == u[1]:
                assert v[0] > u[0]

    def test_multigraph(self):
        """Tests for creating a triangular lattice multigraph."""
        G = nx.triangular_lattice_graph(3, 4, create_using=nx.Graph())
        H = nx.triangular_lattice_graph(3, 4, create_using=nx.MultiGraph())
        assert list(H.edges()) == list(G.edges())

    def test_periodic(self):
        G = nx.triangular_lattice_graph(4, 6, periodic=True)
        assert len(G) == 12
        assert G.size() == 36
        assert len([n for n, d in G.degree() if d != 6]) == 0
        G = nx.triangular_lattice_graph(5, 7, periodic=True)
        TLG = nx.triangular_lattice_graph
        pytest.raises(nx.NetworkXError, TLG, 2, 4, periodic=True)
        pytest.raises(nx.NetworkXError, TLG, 4, 4, periodic=True)
        pytest.raises(nx.NetworkXError, TLG, 2, 6, periodic=True)