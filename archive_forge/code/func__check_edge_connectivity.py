import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def _check_edge_connectivity(G):
    """
    Helper - generates all k-edge-components using the aux graph.  Checks the
    both local and subgraph edge connectivity of each cc. Also checks that
    alternate methods of computing the k-edge-ccs generate the same result.
    """
    aux_graph = EdgeComponentAuxGraph.construct(G)
    memo = {}
    for k in it.count(1):
        ccs_local = fset(aux_graph.k_edge_components(k))
        ccs_subgraph = fset(aux_graph.k_edge_subgraphs(k))
        _assert_local_cc_edge_connectivity(G, ccs_local, k, memo)
        _assert_subgraph_edge_connectivity(G, ccs_subgraph, k)
        if k == 1 or (k == 2 and (not G.is_directed())):
            assert ccs_local == ccs_subgraph, 'Subgraphs and components should be the same when k == 1 or (k == 2 and not G.directed())'
        if G.is_directed():
            if k == 1:
                alt_sccs = fset(nx.strongly_connected_components(G))
                assert alt_sccs == ccs_local, 'k=1 failed alt'
                assert alt_sccs == ccs_subgraph, 'k=1 failed alt'
        elif k == 1:
            alt_ccs = fset(nx.connected_components(G))
            assert alt_ccs == ccs_local, 'k=1 failed alt'
            assert alt_ccs == ccs_subgraph, 'k=1 failed alt'
        elif k == 2:
            alt_bridge_ccs = fset(bridge_components(G))
            assert alt_bridge_ccs == ccs_local, 'k=2 failed alt'
            assert alt_bridge_ccs == ccs_subgraph, 'k=2 failed alt'
        alt_subgraph_ccs = fset([set(C.nodes()) for C in general_k_edge_subgraphs(G, k=k)])
        assert alt_subgraph_ccs == ccs_subgraph, 'alt subgraph method failed'
        if k > 2 and all((len(cc) == 1 for cc in ccs_local)):
            break