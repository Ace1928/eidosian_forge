import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
class TestAllGraphTypesEdgeCases:

    @pytest.mark.parametrize('graph_type', (nx.Graph, nx.MultiGraph, nx.DiGraph))
    def test_both_graphs_empty(self, graph_type):
        G = graph_type()
        H = graph_type()
        assert vf2pp_isomorphism(G, H) is None
        G.add_node(0)
        assert vf2pp_isomorphism(G, H) is None
        assert vf2pp_isomorphism(H, G) is None
        H.add_node(0)
        assert vf2pp_isomorphism(G, H) == {0: 0}

    @pytest.mark.parametrize('graph_type', (nx.Graph, nx.MultiGraph, nx.DiGraph))
    def test_first_graph_empty(self, graph_type):
        G = graph_type()
        H = graph_type([(0, 1)])
        assert vf2pp_isomorphism(G, H) is None

    @pytest.mark.parametrize('graph_type', (nx.Graph, nx.MultiGraph, nx.DiGraph))
    def test_second_graph_empty(self, graph_type):
        G = graph_type([(0, 1)])
        H = graph_type()
        assert vf2pp_isomorphism(G, H) is None