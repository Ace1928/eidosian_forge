import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestChainsOfViews:

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(9)
        cls.DG = nx.path_graph(9, create_using=nx.DiGraph())
        cls.MG = nx.path_graph(9, create_using=nx.MultiGraph())
        cls.MDG = nx.path_graph(9, create_using=nx.MultiDiGraph())
        cls.Gv = nx.to_undirected(cls.DG)
        cls.DGv = nx.to_directed(cls.G)
        cls.MGv = nx.to_undirected(cls.MDG)
        cls.MDGv = nx.to_directed(cls.MG)
        cls.Rv = cls.DG.reverse()
        cls.MRv = cls.MDG.reverse()
        cls.graphs = [cls.G, cls.DG, cls.MG, cls.MDG, cls.Gv, cls.DGv, cls.MGv, cls.MDGv, cls.Rv, cls.MRv]
        for G in cls.graphs:
            (G.edges, G.nodes, G.degree)

    def test_pickle(self):
        import pickle
        for G in self.graphs:
            H = pickle.loads(pickle.dumps(G, -1))
            assert edges_equal(H.edges, G.edges)
            assert nodes_equal(H.nodes, G.nodes)

    def test_subgraph_of_subgraph(self):
        SGv = nx.subgraph(self.G, range(3, 7))
        SDGv = nx.subgraph(self.DG, range(3, 7))
        SMGv = nx.subgraph(self.MG, range(3, 7))
        SMDGv = nx.subgraph(self.MDG, range(3, 7))
        for G in self.graphs + [SGv, SDGv, SMGv, SMDGv]:
            SG = nx.induced_subgraph(G, [4, 5, 6])
            assert list(SG) == [4, 5, 6]
            SSG = SG.subgraph([6, 7])
            assert list(SSG) == [6]
            assert SSG._graph is G

    def test_restricted_induced_subgraph_chains(self):
        """Test subgraph chains that both restrict and show nodes/edges.

        A restricted_view subgraph should allow induced subgraphs using
        G.subgraph that automagically without a chain (meaning the result
        is a subgraph view of the original graph not a subgraph-of-subgraph.
        """
        hide_nodes = [3, 4, 5]
        hide_edges = [(6, 7)]
        RG = nx.restricted_view(self.G, hide_nodes, hide_edges)
        nodes = [4, 5, 6, 7, 8]
        SG = nx.induced_subgraph(RG, nodes)
        SSG = RG.subgraph(nodes)
        assert RG._graph is self.G
        assert SSG._graph is self.G
        assert SG._graph is RG
        assert edges_equal(SG.edges, SSG.edges)
        CG = self.G.copy()
        CG.remove_nodes_from(hide_nodes)
        CG.remove_edges_from(hide_edges)
        assert edges_equal(CG.edges(nodes), SSG.edges)
        CG.remove_nodes_from([0, 1, 2, 3])
        assert edges_equal(CG.edges, SSG.edges)
        SSSG = self.G.subgraph(nodes)
        RSG = nx.restricted_view(SSSG, hide_nodes, hide_edges)
        assert RSG._graph is not self.G
        assert edges_equal(RSG.edges, CG.edges)

    def test_subgraph_copy(self):
        for origG in self.graphs:
            G = nx.Graph(origG)
            SG = G.subgraph([4, 5, 6])
            H = SG.copy()
            assert type(G) == type(H)

    def test_subgraph_todirected(self):
        SG = nx.induced_subgraph(self.G, [4, 5, 6])
        SSG = SG.to_directed()
        assert sorted(SSG) == [4, 5, 6]
        assert sorted(SSG.edges) == [(4, 5), (5, 4), (5, 6), (6, 5)]

    def test_subgraph_toundirected(self):
        SG = nx.induced_subgraph(self.G, [4, 5, 6])
        SSG = SG.to_undirected()
        assert list(SSG) == [4, 5, 6]
        assert sorted(SSG.edges) == [(4, 5), (5, 6)]

    def test_reverse_subgraph_toundirected(self):
        G = self.DG.reverse(copy=False)
        SG = G.subgraph([4, 5, 6])
        SSG = SG.to_undirected()
        assert list(SSG) == [4, 5, 6]
        assert sorted(SSG.edges) == [(4, 5), (5, 6)]

    def test_reverse_reverse_copy(self):
        G = self.DG.reverse(copy=False)
        H = G.reverse(copy=True)
        assert H.nodes == self.DG.nodes
        assert H.edges == self.DG.edges
        G = self.MDG.reverse(copy=False)
        H = G.reverse(copy=True)
        assert H.nodes == self.MDG.nodes
        assert H.edges == self.MDG.edges

    def test_subgraph_edgesubgraph_toundirected(self):
        G = self.G.copy()
        SG = G.subgraph([4, 5, 6])
        SSG = SG.edge_subgraph([(4, 5), (5, 4)])
        USSG = SSG.to_undirected()
        assert list(USSG) == [4, 5]
        assert sorted(USSG.edges) == [(4, 5)]

    def test_copy_subgraph(self):
        G = self.G.copy()
        SG = G.subgraph([4, 5, 6])
        CSG = SG.copy(as_view=True)
        DCSG = SG.copy(as_view=False)
        assert hasattr(CSG, '_graph')
        assert not hasattr(DCSG, '_graph')

    def test_copy_disubgraph(self):
        G = self.DG.copy()
        SG = G.subgraph([4, 5, 6])
        CSG = SG.copy(as_view=True)
        DCSG = SG.copy(as_view=False)
        assert hasattr(CSG, '_graph')
        assert not hasattr(DCSG, '_graph')

    def test_copy_multidisubgraph(self):
        G = self.MDG.copy()
        SG = G.subgraph([4, 5, 6])
        CSG = SG.copy(as_view=True)
        DCSG = SG.copy(as_view=False)
        assert hasattr(CSG, '_graph')
        assert not hasattr(DCSG, '_graph')

    def test_copy_multisubgraph(self):
        G = self.MG.copy()
        SG = G.subgraph([4, 5, 6])
        CSG = SG.copy(as_view=True)
        DCSG = SG.copy(as_view=False)
        assert hasattr(CSG, '_graph')
        assert not hasattr(DCSG, '_graph')

    def test_copy_of_view(self):
        G = nx.MultiGraph(self.MGv)
        assert G.__class__.__name__ == 'MultiGraph'
        G = G.copy(as_view=True)
        assert G.__class__.__name__ == 'MultiGraph'

    def test_subclass(self):

        class MyGraph(nx.DiGraph):

            def my_method(self):
                return 'me'

            def to_directed_class(self):
                return MyGraph()
        for origG in self.graphs:
            G = MyGraph(origG)
            SG = G.subgraph([4, 5, 6])
            H = SG.copy()
            assert SG.my_method() == 'me'
            assert H.my_method() == 'me'
            assert 3 not in H or 3 in SG