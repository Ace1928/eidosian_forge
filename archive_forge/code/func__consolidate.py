from collections import defaultdict
from itertools import combinations
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import edmonds_karp
from networkx.utils import not_implemented_for
def _consolidate(sets, k):
    """Merge sets that share k or more elements.

    See: http://rosettacode.org/wiki/Set_consolidation

    The iterative python implementation posted there is
    faster than this because of the overhead of building a
    Graph and calling nx.connected_components, but it's not
    clear for us if we can use it in NetworkX because there
    is no licence for the code.

    """
    G = nx.Graph()
    nodes = dict(enumerate(sets))
    G.add_nodes_from(nodes)
    G.add_edges_from(((u, v) for u, v in combinations(nodes, 2) if len(nodes[u] & nodes[v]) >= k))
    for component in nx.connected_components(G):
        yield set.union(*[nodes[n] for n in component])