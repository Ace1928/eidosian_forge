import pytest
import networkx as nx
from networkx.utils import edges_equal
class TestMultiGraphView(TestSubGraphView):
    gview = staticmethod(nx.subgraph_view)
    graph = nx.MultiGraph
    hide_edges_filter = staticmethod(nx.filters.hide_multiedges)
    show_edges_filter = staticmethod(nx.filters.show_multiedges)

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(9, create_using=cls.graph())
        multiedges = {(2, 3, 4), (2, 3, 5)}
        cls.G.add_edges_from(multiedges)
        cls.hide_edges_w_hide_nodes = {(3, 4, 0), (4, 5, 0), (5, 6, 0)}

    def test_hidden_edges(self):
        hide_edges = [(2, 3, 4), (2, 3, 3), (8, 7, 0), (222, 223, 0)]
        edges_gone = self.hide_edges_filter(hide_edges)
        G = self.gview(self.G, filter_edge=edges_gone)
        assert self.G.nodes == G.nodes
        if G.is_directed():
            assert self.G.edges - G.edges == {(2, 3, 4)}
            assert list(G[3]) == [4]
            assert list(G[2]) == [3]
            assert list(G.pred[3]) == [2]
            assert list(G.pred[2]) == [1]
            assert G.size() == 9
        else:
            assert self.G.edges - G.edges == {(2, 3, 4), (7, 8, 0)}
            assert list(G[3]) == [2, 4]
            assert list(G[2]) == [1, 3]
            assert G.size() == 8
        assert G.degree(3) == 3
        pytest.raises(KeyError, G.__getitem__, 221)
        pytest.raises(KeyError, G.__getitem__, 222)

    def test_shown_edges(self):
        show_edges = [(2, 3, 4), (2, 3, 3), (8, 7, 0), (222, 223, 0)]
        edge_subgraph = self.show_edges_filter(show_edges)
        G = self.gview(self.G, filter_edge=edge_subgraph)
        assert self.G.nodes == G.nodes
        if G.is_directed():
            assert G.edges == {(2, 3, 4)}
            assert list(G[3]) == []
            assert list(G.pred[3]) == [2]
            assert list(G.pred[2]) == []
            assert G.size() == 1
        else:
            assert G.edges == {(2, 3, 4), (7, 8, 0)}
            assert G.size() == 2
            assert list(G[3]) == [2]
        assert G.degree(3) == 1
        assert list(G[2]) == [3]
        pytest.raises(KeyError, G.__getitem__, 221)
        pytest.raises(KeyError, G.__getitem__, 222)