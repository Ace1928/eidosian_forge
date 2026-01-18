import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
def _check_separating_sets(G):
    for cc in nx.connected_components(G):
        if len(cc) < 3:
            continue
        Gc = G.subgraph(cc)
        node_conn = nx.node_connectivity(Gc)
        all_cuts = nx.all_node_cuts(Gc)
        for cut in itertools.islice(all_cuts, MAX_CUTSETS_TO_TEST):
            assert node_conn == len(cut)
            assert not nx.is_connected(nx.restricted_view(G, cut, []))