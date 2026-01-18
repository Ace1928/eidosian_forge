import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
from networkx.algorithms.flow import (
class TestMaxflowLargeGraph:

    def test_complete_graph(self):
        N = 50
        G = nx.complete_graph(N)
        nx.set_edge_attributes(G, 5, 'capacity')
        R = build_residual_network(G, 'capacity')
        kwargs = {'residual': R}
        for flow_func in flow_funcs:
            kwargs['flow_func'] = flow_func
            errmsg = f'Assertion failed in function: {flow_func.__name__}'
            flow_value = nx.maximum_flow_value(G, 1, 2, **kwargs)
            assert flow_value == 5 * (N - 1), errmsg

    def test_pyramid(self):
        N = 10
        G = gen_pyramid(N)
        R = build_residual_network(G, 'capacity')
        kwargs = {'residual': R}
        for flow_func in flow_funcs:
            kwargs['flow_func'] = flow_func
            errmsg = f'Assertion failed in function: {flow_func.__name__}'
            flow_value = nx.maximum_flow_value(G, (0, 0), 't', **kwargs)
            assert flow_value == pytest.approx(1.0, abs=1e-07)

    def test_gl1(self):
        G = read_graph('gl1')
        s = 1
        t = len(G)
        R = build_residual_network(G, 'capacity')
        kwargs = {'residual': R}
        flow_func = flow_funcs[0]
        validate_flows(G, s, t, 156545, flow_func(G, s, t, **kwargs), flow_func)

    @pytest.mark.slow
    def test_gw1(self):
        G = read_graph('gw1')
        s = 1
        t = len(G)
        R = build_residual_network(G, 'capacity')
        kwargs = {'residual': R}
        for flow_func in flow_funcs:
            validate_flows(G, s, t, 1202018, flow_func(G, s, t, **kwargs), flow_func)

    def test_wlm3(self):
        G = read_graph('wlm3')
        s = 1
        t = len(G)
        R = build_residual_network(G, 'capacity')
        kwargs = {'residual': R}
        flow_func = flow_funcs[0]
        validate_flows(G, s, t, 11875108, flow_func(G, s, t, **kwargs), flow_func)

    def test_preflow_push_global_relabel(self):
        G = read_graph('gw1')
        R = preflow_push(G, 1, len(G), global_relabel_freq=50)
        assert R.graph['flow_value'] == 1202018