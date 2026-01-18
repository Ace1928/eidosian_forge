import pytest
import networkx as nx
from networkx.classes import Graph, MultiDiGraph
from networkx.generators.directed import (
class TestGeneratorsDirected:

    def test_smoke_test_random_graphs(self):
        gn_graph(100)
        gnr_graph(100, 0.5)
        gnc_graph(100)
        scale_free_graph(100)
        gn_graph(100, seed=42)
        gnr_graph(100, 0.5, seed=42)
        gnc_graph(100, seed=42)
        scale_free_graph(100, seed=42)

    def test_create_using_keyword_arguments(self):
        pytest.raises(nx.NetworkXError, gn_graph, 100, create_using=Graph())
        pytest.raises(nx.NetworkXError, gnr_graph, 100, 0.5, create_using=Graph())
        pytest.raises(nx.NetworkXError, gnc_graph, 100, create_using=Graph())
        G = gn_graph(100, seed=1)
        MG = gn_graph(100, create_using=MultiDiGraph(), seed=1)
        assert sorted(G.edges()) == sorted(MG.edges())
        G = gnr_graph(100, 0.5, seed=1)
        MG = gnr_graph(100, 0.5, create_using=MultiDiGraph(), seed=1)
        assert sorted(G.edges()) == sorted(MG.edges())
        G = gnc_graph(100, seed=1)
        MG = gnc_graph(100, create_using=MultiDiGraph(), seed=1)
        assert sorted(G.edges()) == sorted(MG.edges())
        G = scale_free_graph(100, alpha=0.3, beta=0.4, gamma=0.3, delta_in=0.3, delta_out=0.1, initial_graph=nx.cycle_graph(4, create_using=MultiDiGraph), seed=1)
        pytest.raises(ValueError, scale_free_graph, 100, 0.5, 0.4, 0.3)
        pytest.raises(ValueError, scale_free_graph, 100, alpha=-0.3)
        pytest.raises(ValueError, scale_free_graph, 100, beta=-0.3)
        pytest.raises(ValueError, scale_free_graph, 100, gamma=-0.3)

    def test_parameters(self):
        G = nx.DiGraph()
        G.add_node(0)

        def kernel(x):
            return x
        assert nx.is_isomorphic(gn_graph(1), G)
        assert nx.is_isomorphic(gn_graph(1, kernel=kernel), G)
        assert nx.is_isomorphic(gnc_graph(1), G)
        assert nx.is_isomorphic(gnr_graph(1, 0.5), G)