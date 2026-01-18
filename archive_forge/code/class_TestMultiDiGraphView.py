import pytest
import networkx as nx
from networkx.utils import edges_equal
class TestMultiDiGraphView(TestMultiGraphView, TestSubDiGraphView):
    gview = staticmethod(nx.subgraph_view)
    graph = nx.MultiDiGraph
    hide_edges_filter = staticmethod(nx.filters.hide_multidiedges)
    show_edges_filter = staticmethod(nx.filters.show_multidiedges)
    hide_edges = [(2, 3, 0), (8, 7, 0), (222, 223, 0)]
    excluded = {(2, 3, 0), (3, 4, 0), (4, 5, 0), (5, 6, 0)}

    def test_inout_degree(self):
        edges_gone = self.hide_edges_filter(self.hide_edges)
        hide_nodes = [4, 5, 111]
        nodes_gone = nx.filters.hide_nodes(hide_nodes)
        G = self.gview(self.G, filter_node=nodes_gone, filter_edge=edges_gone)
        assert G.degree(2) == 3
        assert G.out_degree(2) == 2
        assert G.in_degree(2) == 1
        assert G.size() == 6