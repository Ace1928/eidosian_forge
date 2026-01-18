from collections import defaultdict
from itertools import combinations
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from networkx.utils import not_implemented_for
def _generate_partition(G, cuts, k):

    def has_nbrs_in_partition(G, node, partition):
        return any((n in partition for n in G[node]))
    components = []
    nodes = {n for n, d in G.degree() if d > k} - {n for cut in cuts for n in cut}
    H = G.subgraph(nodes)
    for cc in nx.connected_components(H):
        component = set(cc)
        for cut in cuts:
            for node in cut:
                if has_nbrs_in_partition(G, node, cc):
                    component.add(node)
        if len(component) < G.order():
            components.append(component)
    yield from _consolidate(components, k + 1)