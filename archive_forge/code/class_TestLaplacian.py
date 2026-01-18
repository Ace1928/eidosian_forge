import pytest
import networkx as nx
from networkx.generators.degree_seq import havel_hakimi_graph
from networkx.generators.expanders import margulis_gabber_galil_graph
class TestLaplacian:

    @classmethod
    def setup_class(cls):
        deg = [3, 2, 2, 1, 0]
        cls.G = havel_hakimi_graph(deg)
        cls.WG = nx.Graph(((u, v, {'weight': 0.5, 'other': 0.3}) for u, v in cls.G.edges()))
        cls.WG.add_node(4)
        cls.MG = nx.MultiGraph(cls.G)
        cls.Gsl = cls.G.copy()
        for node in cls.Gsl.nodes():
            cls.Gsl.add_edge(node, node)

    def test_laplacian(self):
        """Graph Laplacian"""
        NL = np.array([[3, -1, -1, -1, 0], [-1, 2, -1, 0, 0], [-1, -1, 2, 0, 0], [-1, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
        WL = 0.5 * NL
        OL = 0.3 * NL
        np.testing.assert_equal(nx.laplacian_matrix(self.G).todense(), NL)
        np.testing.assert_equal(nx.laplacian_matrix(self.MG).todense(), NL)
        np.testing.assert_equal(nx.laplacian_matrix(self.G, nodelist=[0, 1]).todense(), np.array([[1, -1], [-1, 1]]))
        np.testing.assert_equal(nx.laplacian_matrix(self.WG).todense(), WL)
        np.testing.assert_equal(nx.laplacian_matrix(self.WG, weight=None).todense(), NL)
        np.testing.assert_equal(nx.laplacian_matrix(self.WG, weight='other').todense(), OL)

    def test_normalized_laplacian(self):
        """Generalized Graph Laplacian"""
        G = np.array([[1.0, -0.408, -0.408, -0.577, 0.0], [-0.408, 1.0, -0.5, 0.0, 0.0], [-0.408, -0.5, 1.0, 0.0, 0.0], [-0.577, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        GL = np.array([[1.0, -0.408, -0.408, -0.577, 0.0], [-0.408, 1.0, -0.5, 0.0, 0.0], [-0.408, -0.5, 1.0, 0.0, 0.0], [-0.577, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        Lsl = np.array([[0.75, -0.2887, -0.2887, -0.3536, 0.0], [-0.2887, 0.6667, -0.3333, 0.0, 0.0], [-0.2887, -0.3333, 0.6667, 0.0, 0.0], [-0.3536, 0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
        np.testing.assert_almost_equal(nx.normalized_laplacian_matrix(self.G, nodelist=range(5)).todense(), G, decimal=3)
        np.testing.assert_almost_equal(nx.normalized_laplacian_matrix(self.G).todense(), GL, decimal=3)
        np.testing.assert_almost_equal(nx.normalized_laplacian_matrix(self.MG).todense(), GL, decimal=3)
        np.testing.assert_almost_equal(nx.normalized_laplacian_matrix(self.WG).todense(), GL, decimal=3)
        np.testing.assert_almost_equal(nx.normalized_laplacian_matrix(self.WG, weight='other').todense(), GL, decimal=3)
        np.testing.assert_almost_equal(nx.normalized_laplacian_matrix(self.Gsl).todense(), Lsl, decimal=3)