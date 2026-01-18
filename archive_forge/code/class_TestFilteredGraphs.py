import pickle
import pytest
import networkx as nx
class TestFilteredGraphs:

    def setup_method(self):
        self.Graphs = [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]

    def test_hide_show_nodes(self):
        SubGraph = nx.subgraph_view
        for Graph in self.Graphs:
            G = nx.path_graph(4, Graph)
            SG = G.subgraph([2, 3])
            RG = SubGraph(G, filter_node=nx.filters.hide_nodes([0, 1]))
            assert SG.nodes == RG.nodes
            assert SG.edges == RG.edges
            SGC = SG.copy()
            RGC = RG.copy()
            assert SGC.nodes == RGC.nodes
            assert SGC.edges == RGC.edges

    def test_str_repr(self):
        SubGraph = nx.subgraph_view
        for Graph in self.Graphs:
            G = nx.path_graph(4, Graph)
            SG = G.subgraph([2, 3])
            RG = SubGraph(G, filter_node=nx.filters.hide_nodes([0, 1]))
            str(SG.adj)
            str(RG.adj)
            repr(SG.adj)
            repr(RG.adj)
            str(SG.adj[2])
            str(RG.adj[2])
            repr(SG.adj[2])
            repr(RG.adj[2])

    def test_copy(self):
        SubGraph = nx.subgraph_view
        for Graph in self.Graphs:
            G = nx.path_graph(4, Graph)
            SG = G.subgraph([2, 3])
            RG = SubGraph(G, filter_node=nx.filters.hide_nodes([0, 1]))
            RsG = SubGraph(G, filter_node=nx.filters.show_nodes([2, 3]))
            assert G.adj.copy() == G.adj
            assert G.adj[2].copy() == G.adj[2]
            assert SG.adj.copy() == SG.adj
            assert SG.adj[2].copy() == SG.adj[2]
            assert RG.adj.copy() == RG.adj
            assert RG.adj[2].copy() == RG.adj[2]
            assert RsG.adj.copy() == RsG.adj
            assert RsG.adj[2].copy() == RsG.adj[2]