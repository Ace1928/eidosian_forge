import pytest
import networkx as nx
from networkx.utils import edges_equal
class TestSubGraphView:
    gview = staticmethod(nx.subgraph_view)
    graph = nx.Graph
    hide_edges_filter = staticmethod(nx.filters.hide_edges)
    show_edges_filter = staticmethod(nx.filters.show_edges)

    @classmethod
    def setup_class(cls):
        cls.G = nx.path_graph(9, create_using=cls.graph())
        cls.hide_edges_w_hide_nodes = {(3, 4), (4, 5), (5, 6)}

    def test_hidden_nodes(self):
        hide_nodes = [4, 5, 111]
        nodes_gone = nx.filters.hide_nodes(hide_nodes)
        gview = self.gview
        G = gview(self.G, filter_node=nodes_gone)
        assert self.G.nodes - G.nodes == {4, 5}
        assert self.G.edges - G.edges == self.hide_edges_w_hide_nodes
        if G.is_directed():
            assert list(G[3]) == []
            assert list(G[2]) == [3]
        else:
            assert list(G[3]) == [2]
            assert set(G[2]) == {1, 3}
        pytest.raises(KeyError, G.__getitem__, 4)
        pytest.raises(KeyError, G.__getitem__, 112)
        pytest.raises(KeyError, G.__getitem__, 111)
        assert G.degree(3) == (3 if G.is_multigraph() else 1)
        assert G.size() == (7 if G.is_multigraph() else 5)

    def test_hidden_edges(self):
        hide_edges = [(2, 3), (8, 7), (222, 223)]
        edges_gone = self.hide_edges_filter(hide_edges)
        gview = self.gview
        G = gview(self.G, filter_edge=edges_gone)
        assert self.G.nodes == G.nodes
        if G.is_directed():
            assert self.G.edges - G.edges == {(2, 3)}
            assert list(G[2]) == []
            assert list(G.pred[3]) == []
            assert list(G.pred[2]) == [1]
            assert G.size() == 7
        else:
            assert self.G.edges - G.edges == {(2, 3), (7, 8)}
            assert list(G[2]) == [1]
            assert G.size() == 6
        assert list(G[3]) == [4]
        pytest.raises(KeyError, G.__getitem__, 221)
        pytest.raises(KeyError, G.__getitem__, 222)
        assert G.degree(3) == 1

    def test_shown_node(self):
        induced_subgraph = nx.filters.show_nodes([2, 3, 111])
        gview = self.gview
        G = gview(self.G, filter_node=induced_subgraph)
        assert set(G.nodes) == {2, 3}
        if G.is_directed():
            assert list(G[3]) == []
        else:
            assert list(G[3]) == [2]
        assert list(G[2]) == [3]
        pytest.raises(KeyError, G.__getitem__, 4)
        pytest.raises(KeyError, G.__getitem__, 112)
        pytest.raises(KeyError, G.__getitem__, 111)
        assert G.degree(3) == (3 if G.is_multigraph() else 1)
        assert G.size() == (3 if G.is_multigraph() else 1)

    def test_shown_edges(self):
        show_edges = [(2, 3), (8, 7), (222, 223)]
        edge_subgraph = self.show_edges_filter(show_edges)
        G = self.gview(self.G, filter_edge=edge_subgraph)
        assert self.G.nodes == G.nodes
        if G.is_directed():
            assert G.edges == {(2, 3)}
            assert list(G[3]) == []
            assert list(G[2]) == [3]
            assert list(G.pred[3]) == [2]
            assert list(G.pred[2]) == []
            assert G.size() == 1
        else:
            assert G.edges == {(2, 3), (7, 8)}
            assert list(G[3]) == [2]
            assert list(G[2]) == [3]
            assert G.size() == 2
        pytest.raises(KeyError, G.__getitem__, 221)
        pytest.raises(KeyError, G.__getitem__, 222)
        assert G.degree(3) == 1