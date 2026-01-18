import pytest
import networkx as nx
from networkx.algorithms.flow import (
class TestCutoff:

    def test_cutoff(self):
        k = 5
        p = 1000
        G = nx.DiGraph()
        for i in range(k):
            G.add_edge('s', (i, 0), capacity=2)
            nx.add_path(G, ((i, j) for j in range(p)), capacity=2)
            G.add_edge((i, p - 1), 't', capacity=2)
        R = shortest_augmenting_path(G, 's', 't', two_phase=True, cutoff=k)
        assert k <= R.graph['flow_value'] <= 2 * k
        R = shortest_augmenting_path(G, 's', 't', two_phase=False, cutoff=k)
        assert k <= R.graph['flow_value'] <= 2 * k
        R = edmonds_karp(G, 's', 't', cutoff=k)
        assert k <= R.graph['flow_value'] <= 2 * k
        R = dinitz(G, 's', 't', cutoff=k)
        assert k <= R.graph['flow_value'] <= 2 * k
        R = boykov_kolmogorov(G, 's', 't', cutoff=k)
        assert k <= R.graph['flow_value'] <= 2 * k

    def test_complete_graph_cutoff(self):
        G = nx.complete_graph(5)
        nx.set_edge_attributes(G, {(u, v): 1 for u, v in G.edges()}, 'capacity')
        for flow_func in [shortest_augmenting_path, edmonds_karp, dinitz, boykov_kolmogorov]:
            for cutoff in [3, 2, 1]:
                result = nx.maximum_flow_value(G, 0, 4, flow_func=flow_func, cutoff=cutoff)
                assert cutoff == result, f'cutoff error in {flow_func.__name__}'