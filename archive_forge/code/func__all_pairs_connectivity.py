import itertools as it
import pytest
import networkx as nx
from networkx.algorithms.connectivity import EdgeComponentAuxGraph, bridge_components
from networkx.algorithms.connectivity.edge_kcomponents import general_k_edge_subgraphs
from networkx.utils import pairwise
def _all_pairs_connectivity(G, cc, k, memo):
    for u, v in it.combinations(cc, 2):
        connectivity = _memo_connectivity(G, u, v, memo)
        if G.is_directed():
            connectivity = min(connectivity, _memo_connectivity(G, v, u, memo))
        assert connectivity >= k