import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def _assert_local_cc_edge_connectivity(G, ccs_local, k, memo):
    """
    tests properties of k-edge-connected components

    the local edge connectivity between each pair of nodes in the original
    graph should be no less than k unless the cc is a single node.
    """
    for cc in ccs_local:
        if len(cc) > 1:
            C = G.subgraph(cc)
            connectivity = nx.edge_connectivity(C)
            if connectivity < k:
                _all_pairs_connectivity(G, cc, k, memo)