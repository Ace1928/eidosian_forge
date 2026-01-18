import pytest
import networkx as nx
from networkx.generators.classic import barbell_graph, cycle_graph, path_graph
from networkx.utils import graphs_equal
class TestConvertNumpyArray:

    def setup_method(self):
        self.G1 = barbell_graph(10, 3)
        self.G2 = cycle_graph(10, create_using=nx.DiGraph)
        self.G3 = self.create_weighted(nx.Graph())
        self.G4 = self.create_weighted(nx.DiGraph())

    def create_weighted(self, G):
        g = cycle_graph(4)
        G.add_nodes_from(g)
        G.add_weighted_edges_from(((u, v, 10 + u) for u, v in g.edges()))
        return G

    def assert_equal(self, G1, G2):
        assert sorted(G1.nodes()) == sorted(G2.nodes())
        assert sorted(G1.edges()) == sorted(G2.edges())

    def identity_conversion(self, G, A, create_using):
        assert A.sum() > 0
        GG = nx.from_numpy_array(A, create_using=create_using)
        self.assert_equal(G, GG)
        GW = nx.to_networkx_graph(A, create_using=create_using)
        self.assert_equal(G, GW)
        GI = nx.empty_graph(0, create_using).__class__(A)
        self.assert_equal(G, GI)

    def test_shape(self):
        """Conversion from non-square array."""
        A = np.array([[1, 2, 3], [4, 5, 6]])
        pytest.raises(nx.NetworkXError, nx.from_numpy_array, A)

    def test_identity_graph_array(self):
        """Conversion from graph to array to graph."""
        A = nx.to_numpy_array(self.G1)
        self.identity_conversion(self.G1, A, nx.Graph())

    def test_identity_digraph_array(self):
        """Conversion from digraph to array to digraph."""
        A = nx.to_numpy_array(self.G2)
        self.identity_conversion(self.G2, A, nx.DiGraph())

    def test_identity_weighted_graph_array(self):
        """Conversion from weighted graph to array to weighted graph."""
        A = nx.to_numpy_array(self.G3)
        self.identity_conversion(self.G3, A, nx.Graph())

    def test_identity_weighted_digraph_array(self):
        """Conversion from weighted digraph to array to weighted digraph."""
        A = nx.to_numpy_array(self.G4)
        self.identity_conversion(self.G4, A, nx.DiGraph())

    def test_nodelist(self):
        """Conversion from graph to array to graph with nodelist."""
        P4 = path_graph(4)
        P3 = path_graph(3)
        nodelist = list(P3)
        A = nx.to_numpy_array(P4, nodelist=nodelist)
        GA = nx.Graph(A)
        self.assert_equal(GA, P3)
        nodelist += [nodelist[0]]
        pytest.raises(nx.NetworkXError, nx.to_numpy_array, P3, nodelist=nodelist)
        nodelist = [-1, 0, 1]
        with pytest.raises(nx.NetworkXError, match=f'Nodes {nodelist - P3.nodes} in nodelist is not in G'):
            nx.to_numpy_array(P3, nodelist=nodelist)

    def test_weight_keyword(self):
        WP4 = nx.Graph()
        WP4.add_edges_from(((n, n + 1, {'weight': 0.5, 'other': 0.3}) for n in range(3)))
        P4 = path_graph(4)
        A = nx.to_numpy_array(P4)
        np.testing.assert_equal(A, nx.to_numpy_array(WP4, weight=None))
        np.testing.assert_equal(0.5 * A, nx.to_numpy_array(WP4))
        np.testing.assert_equal(0.3 * A, nx.to_numpy_array(WP4, weight='other'))

    def test_from_numpy_array_type(self):
        A = np.array([[1]])
        G = nx.from_numpy_array(A)
        assert type(G[0][0]['weight']) == int
        A = np.array([[1]]).astype(float)
        G = nx.from_numpy_array(A)
        assert type(G[0][0]['weight']) == float
        A = np.array([[1]]).astype(str)
        G = nx.from_numpy_array(A)
        assert type(G[0][0]['weight']) == str
        A = np.array([[1]]).astype(bool)
        G = nx.from_numpy_array(A)
        assert type(G[0][0]['weight']) == bool
        A = np.array([[1]]).astype(complex)
        G = nx.from_numpy_array(A)
        assert type(G[0][0]['weight']) == complex
        A = np.array([[1]]).astype(object)
        pytest.raises(TypeError, nx.from_numpy_array, A)
        A = np.array([[[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]])
        with pytest.raises(nx.NetworkXError, match=f'Input array must be 2D, not {A.ndim}'):
            g = nx.from_numpy_array(A)

    def test_from_numpy_array_dtype(self):
        dt = [('weight', float), ('cost', int)]
        A = np.array([[(1.0, 2)]], dtype=dt)
        G = nx.from_numpy_array(A)
        assert type(G[0][0]['weight']) == float
        assert type(G[0][0]['cost']) == int
        assert G[0][0]['cost'] == 2
        assert G[0][0]['weight'] == 1.0

    def test_from_numpy_array_parallel_edges(self):
        """Tests that the :func:`networkx.from_numpy_array` function
        interprets integer weights as the number of parallel edges when
        creating a multigraph.

        """
        A = np.array([[1, 1], [1, 2]])
        expected = nx.DiGraph()
        edges = [(0, 0), (0, 1), (1, 0)]
        expected.add_weighted_edges_from([(u, v, 1) for u, v in edges])
        expected.add_edge(1, 1, weight=2)
        actual = nx.from_numpy_array(A, parallel_edges=True, create_using=nx.DiGraph)
        assert graphs_equal(actual, expected)
        actual = nx.from_numpy_array(A, parallel_edges=False, create_using=nx.DiGraph)
        assert graphs_equal(actual, expected)
        edges = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 1)]
        expected = nx.MultiDiGraph()
        expected.add_weighted_edges_from([(u, v, 1) for u, v in edges])
        actual = nx.from_numpy_array(A, parallel_edges=True, create_using=nx.MultiDiGraph)
        assert graphs_equal(actual, expected)
        expected = nx.MultiDiGraph()
        expected.add_edges_from(set(edges), weight=1)
        expected[1][1][0]['weight'] = 2
        actual = nx.from_numpy_array(A, parallel_edges=False, create_using=nx.MultiDiGraph)
        assert graphs_equal(actual, expected)

    @pytest.mark.parametrize('dt', (None, int, np.dtype([('weight', 'f8'), ('color', 'i1')])))
    def test_from_numpy_array_no_edge_attr(self, dt):
        A = np.array([[0, 1], [1, 0]], dtype=dt)
        G = nx.from_numpy_array(A, edge_attr=None)
        assert 'weight' not in G.edges[0, 1]
        assert len(G.edges[0, 1]) == 0

    def test_from_numpy_array_multiedge_no_edge_attr(self):
        A = np.array([[0, 2], [2, 0]])
        G = nx.from_numpy_array(A, create_using=nx.MultiDiGraph, edge_attr=None)
        assert all(('weight' not in e for _, e in G[0][1].items()))
        assert len(G[0][1][0]) == 0

    def test_from_numpy_array_custom_edge_attr(self):
        A = np.array([[0, 2], [3, 0]])
        G = nx.from_numpy_array(A, edge_attr='cost')
        assert 'weight' not in G.edges[0, 1]
        assert G.edges[0, 1]['cost'] == 3

    def test_symmetric(self):
        """Tests that a symmetric array has edges added only once to an
        undirected multigraph when using :func:`networkx.from_numpy_array`.

        """
        A = np.array([[0, 1], [1, 0]])
        G = nx.from_numpy_array(A, create_using=nx.MultiGraph)
        expected = nx.MultiGraph()
        expected.add_edge(0, 1, weight=1)
        assert graphs_equal(G, expected)

    def test_dtype_int_graph(self):
        """Test that setting dtype int actually gives an integer array.

        For more information, see GitHub pull request #1363.

        """
        G = nx.complete_graph(3)
        A = nx.to_numpy_array(G, dtype=int)
        assert A.dtype == int

    def test_dtype_int_multigraph(self):
        """Test that setting dtype int actually gives an integer array.

        For more information, see GitHub pull request #1363.

        """
        G = nx.MultiGraph(nx.complete_graph(3))
        A = nx.to_numpy_array(G, dtype=int)
        assert A.dtype == int