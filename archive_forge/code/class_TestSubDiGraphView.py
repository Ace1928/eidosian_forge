import pytest
import networkx as nx
from networkx.utils import edges_equal
class TestSubDiGraphView(TestSubGraphView):
    gview = staticmethod(nx.subgraph_view)
    graph = nx.DiGraph
    hide_edges_filter = staticmethod(nx.filters.hide_diedges)
    show_edges_filter = staticmethod(nx.filters.show_diedges)
    hide_edges = [(2, 3), (8, 7), (222, 223)]
    excluded = {(2, 3), (3, 4), (4, 5), (5, 6)}

    def test_inoutedges(self):
        edges_gone = self.hide_edges_filter(self.hide_edges)
        hide_nodes = [4, 5, 111]
        nodes_gone = nx.filters.hide_nodes(hide_nodes)
        G = self.gview(self.G, filter_node=nodes_gone, filter_edge=edges_gone)
        assert self.G.in_edges - G.in_edges == self.excluded
        assert self.G.out_edges - G.out_edges == self.excluded

    def test_pred(self):
        edges_gone = self.hide_edges_filter(self.hide_edges)
        hide_nodes = [4, 5, 111]
        nodes_gone = nx.filters.hide_nodes(hide_nodes)
        G = self.gview(self.G, filter_node=nodes_gone, filter_edge=edges_gone)
        assert list(G.pred[2]) == [1]
        assert list(G.pred[6]) == []

    def test_inout_degree(self):
        edges_gone = self.hide_edges_filter(self.hide_edges)
        hide_nodes = [4, 5, 111]
        nodes_gone = nx.filters.hide_nodes(hide_nodes)
        G = self.gview(self.G, filter_node=nodes_gone, filter_edge=edges_gone)
        assert G.degree(2) == 1
        assert G.out_degree(2) == 0
        assert G.in_degree(2) == 1
        assert G.size() == 4