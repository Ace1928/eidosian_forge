import itertools
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
import networkx as nx
from networkx.algorithms.approximation import local_node_connectivity
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
def _cliques_heuristic(G, H, k, min_density):
    h_cnumber = nx.core_number(H)
    for i, c_value in enumerate(sorted(set(h_cnumber.values()), reverse=True)):
        cands = {n for n, c in h_cnumber.items() if c == c_value}
        if i == 0:
            overlap = False
        else:
            overlap = set.intersection(*[{x for x in H[n] if x not in cands} for n in cands])
        if overlap and len(overlap) < k:
            SH = H.subgraph(cands | overlap)
        else:
            SH = H.subgraph(cands)
        sh_cnumber = nx.core_number(SH)
        SG = nx.k_core(G.subgraph(SH), k)
        while not (_same(sh_cnumber) and nx.density(SH) >= min_density):
            SH = H.subgraph(SG).copy()
            if len(SH) <= k:
                break
            sh_cnumber = nx.core_number(SH)
            sh_deg = dict(SH.degree())
            min_deg = min(sh_deg.values())
            SH.remove_nodes_from((n for n, d in sh_deg.items() if d == min_deg))
            SG = nx.k_core(G.subgraph(SH), k)
        else:
            yield SG