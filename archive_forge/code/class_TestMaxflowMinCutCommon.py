import pytest
import networkx as nx
from networkx.algorithms.flow import (
class TestMaxflowMinCutCommon:

    def test_graph1(self):
        G = nx.Graph()
        G.add_edge(1, 2, capacity=1.0)
        solnFlows = {1: {2: 1.0}, 2: {1: 1.0}}
        compare_flows_and_cuts(G, 1, 2, solnFlows, 1.0)

    def test_graph2(self):
        G = nx.Graph()
        G.add_edge('x', 'a', capacity=3.0)
        G.add_edge('x', 'b', capacity=1.0)
        G.add_edge('a', 'c', capacity=3.0)
        G.add_edge('b', 'c', capacity=5.0)
        G.add_edge('b', 'd', capacity=4.0)
        G.add_edge('d', 'e', capacity=2.0)
        G.add_edge('c', 'y', capacity=2.0)
        G.add_edge('e', 'y', capacity=3.0)
        H = {'x': {'a': 3, 'b': 1}, 'a': {'c': 3, 'x': 3}, 'b': {'c': 1, 'd': 2, 'x': 1}, 'c': {'a': 3, 'b': 1, 'y': 2}, 'd': {'b': 2, 'e': 2}, 'e': {'d': 2, 'y': 2}, 'y': {'c': 2, 'e': 2}}
        compare_flows_and_cuts(G, 'x', 'y', H, 4.0)

    def test_digraph1(self):
        G = nx.DiGraph()
        G.add_edge('a', 'b', capacity=1000.0)
        G.add_edge('a', 'c', capacity=1000.0)
        G.add_edge('b', 'c', capacity=1.0)
        G.add_edge('b', 'd', capacity=1000.0)
        G.add_edge('c', 'd', capacity=1000.0)
        H = {'a': {'b': 1000.0, 'c': 1000.0}, 'b': {'c': 0, 'd': 1000.0}, 'c': {'d': 1000.0}, 'd': {}}
        compare_flows_and_cuts(G, 'a', 'd', H, 2000.0)

    def test_digraph2(self):
        G = nx.DiGraph()
        G.add_edge('s', 'b', capacity=2)
        G.add_edge('s', 'c', capacity=1)
        G.add_edge('c', 'd', capacity=1)
        G.add_edge('d', 'a', capacity=1)
        G.add_edge('b', 'a', capacity=2)
        G.add_edge('a', 't', capacity=2)
        H = {'s': {'b': 2, 'c': 0}, 'c': {'d': 0}, 'd': {'a': 0}, 'b': {'a': 2}, 'a': {'t': 2}, 't': {}}
        compare_flows_and_cuts(G, 's', 't', H, 2)

    def test_digraph3(self):
        G = nx.DiGraph()
        G.add_edge('s', 'v1', capacity=16.0)
        G.add_edge('s', 'v2', capacity=13.0)
        G.add_edge('v1', 'v2', capacity=10.0)
        G.add_edge('v2', 'v1', capacity=4.0)
        G.add_edge('v1', 'v3', capacity=12.0)
        G.add_edge('v3', 'v2', capacity=9.0)
        G.add_edge('v2', 'v4', capacity=14.0)
        G.add_edge('v4', 'v3', capacity=7.0)
        G.add_edge('v3', 't', capacity=20.0)
        G.add_edge('v4', 't', capacity=4.0)
        H = {'s': {'v1': 12.0, 'v2': 11.0}, 'v2': {'v1': 0, 'v4': 11.0}, 'v1': {'v2': 0, 'v3': 12.0}, 'v3': {'v2': 0, 't': 19.0}, 'v4': {'v3': 7.0, 't': 4.0}, 't': {}}
        compare_flows_and_cuts(G, 's', 't', H, 23.0)

    def test_digraph4(self):
        G = nx.DiGraph()
        G.add_edge('x', 'a', capacity=3.0)
        G.add_edge('x', 'b', capacity=1.0)
        G.add_edge('a', 'c', capacity=3.0)
        G.add_edge('b', 'c', capacity=5.0)
        G.add_edge('b', 'd', capacity=4.0)
        G.add_edge('d', 'e', capacity=2.0)
        G.add_edge('c', 'y', capacity=2.0)
        G.add_edge('e', 'y', capacity=3.0)
        H = {'x': {'a': 2.0, 'b': 1.0}, 'a': {'c': 2.0}, 'b': {'c': 0, 'd': 1.0}, 'c': {'y': 2.0}, 'd': {'e': 1.0}, 'e': {'y': 1.0}, 'y': {}}
        compare_flows_and_cuts(G, 'x', 'y', H, 3.0)

    def test_wikipedia_dinitz_example(self):
        G = nx.DiGraph()
        G.add_edge('s', 1, capacity=10)
        G.add_edge('s', 2, capacity=10)
        G.add_edge(1, 3, capacity=4)
        G.add_edge(1, 4, capacity=8)
        G.add_edge(1, 2, capacity=2)
        G.add_edge(2, 4, capacity=9)
        G.add_edge(3, 't', capacity=10)
        G.add_edge(4, 3, capacity=6)
        G.add_edge(4, 't', capacity=10)
        solnFlows = {1: {2: 0, 3: 4, 4: 6}, 2: {4: 9}, 3: {'t': 9}, 4: {3: 5, 't': 10}, 's': {1: 10, 2: 9}, 't': {}}
        compare_flows_and_cuts(G, 's', 't', solnFlows, 19)

    def test_optional_capacity(self):
        G = nx.DiGraph()
        G.add_edge('x', 'a', spam=3.0)
        G.add_edge('x', 'b', spam=1.0)
        G.add_edge('a', 'c', spam=3.0)
        G.add_edge('b', 'c', spam=5.0)
        G.add_edge('b', 'd', spam=4.0)
        G.add_edge('d', 'e', spam=2.0)
        G.add_edge('c', 'y', spam=2.0)
        G.add_edge('e', 'y', spam=3.0)
        solnFlows = {'x': {'a': 2.0, 'b': 1.0}, 'a': {'c': 2.0}, 'b': {'c': 0, 'd': 1.0}, 'c': {'y': 2.0}, 'd': {'e': 1.0}, 'e': {'y': 1.0}, 'y': {}}
        solnValue = 3.0
        s = 'x'
        t = 'y'
        compare_flows_and_cuts(G, s, t, solnFlows, solnValue, capacity='spam')

    def test_digraph_infcap_edges(self):
        G = nx.DiGraph()
        G.add_edge('s', 'a')
        G.add_edge('s', 'b', capacity=30)
        G.add_edge('a', 'c', capacity=25)
        G.add_edge('b', 'c', capacity=12)
        G.add_edge('a', 't', capacity=60)
        G.add_edge('c', 't')
        H = {'s': {'a': 85, 'b': 12}, 'a': {'c': 25, 't': 60}, 'b': {'c': 12}, 'c': {'t': 37}, 't': {}}
        compare_flows_and_cuts(G, 's', 't', H, 97)
        G = nx.DiGraph()
        G.add_edge('s', 'a', capacity=85)
        G.add_edge('s', 'b', capacity=30)
        G.add_edge('a', 'c')
        G.add_edge('c', 'a')
        G.add_edge('b', 'c', capacity=12)
        G.add_edge('a', 't', capacity=60)
        G.add_edge('c', 't', capacity=37)
        H = {'s': {'a': 85, 'b': 12}, 'a': {'c': 25, 't': 60}, 'c': {'a': 0, 't': 37}, 'b': {'c': 12}, 't': {}}
        compare_flows_and_cuts(G, 's', 't', H, 97)

    def test_digraph_infcap_path(self):
        G = nx.DiGraph()
        G.add_edge('s', 'a')
        G.add_edge('s', 'b', capacity=30)
        G.add_edge('a', 'c')
        G.add_edge('b', 'c', capacity=12)
        G.add_edge('a', 't', capacity=60)
        G.add_edge('c', 't')
        for flow_func in all_funcs:
            pytest.raises(nx.NetworkXUnbounded, flow_func, G, 's', 't')

    def test_graph_infcap_edges(self):
        G = nx.Graph()
        G.add_edge('s', 'a')
        G.add_edge('s', 'b', capacity=30)
        G.add_edge('a', 'c', capacity=25)
        G.add_edge('b', 'c', capacity=12)
        G.add_edge('a', 't', capacity=60)
        G.add_edge('c', 't')
        H = {'s': {'a': 85, 'b': 12}, 'a': {'c': 25, 's': 85, 't': 60}, 'b': {'c': 12, 's': 12}, 'c': {'a': 25, 'b': 12, 't': 37}, 't': {'a': 60, 'c': 37}}
        compare_flows_and_cuts(G, 's', 't', H, 97)

    def test_digraph5(self):
        G = nx.DiGraph()
        G.add_edge('s', 'a', capacity=2)
        G.add_edge('s', 'b', capacity=2)
        G.add_edge('a', 'b', capacity=5)
        G.add_edge('a', 't', capacity=1)
        G.add_edge('b', 'a', capacity=1)
        G.add_edge('b', 't', capacity=3)
        flowSoln = {'a': {'b': 1, 't': 1}, 'b': {'a': 0, 't': 3}, 's': {'a': 2, 'b': 2}, 't': {}}
        compare_flows_and_cuts(G, 's', 't', flowSoln, 4)

    def test_disconnected(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1)], weight='capacity')
        G.remove_node(1)
        assert nx.maximum_flow_value(G, 0, 3) == 0
        flowSoln = {0: {}, 2: {3: 0}, 3: {2: 0}}
        compare_flows_and_cuts(G, 0, 3, flowSoln, 0)

    def test_source_target_not_in_graph(self):
        G = nx.Graph()
        G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1)], weight='capacity')
        G.remove_node(0)
        for flow_func in all_funcs:
            pytest.raises(nx.NetworkXError, flow_func, G, 0, 3)
        G.add_weighted_edges_from([(0, 1, 1), (1, 2, 1), (2, 3, 1)], weight='capacity')
        G.remove_node(3)
        for flow_func in all_funcs:
            pytest.raises(nx.NetworkXError, flow_func, G, 0, 3)

    def test_source_target_coincide(self):
        G = nx.Graph()
        G.add_node(0)
        for flow_func in all_funcs:
            pytest.raises(nx.NetworkXError, flow_func, G, 0, 0)

    def test_multigraphs_raise(self):
        G = nx.MultiGraph()
        M = nx.MultiDiGraph()
        G.add_edges_from([(0, 1), (1, 0)], capacity=True)
        for flow_func in all_funcs:
            pytest.raises(nx.NetworkXError, flow_func, G, 0, 0)