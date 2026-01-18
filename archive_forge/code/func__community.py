from math import log
import networkx as nx
from networkx.utils import not_implemented_for
def _community(G, u, community):
    """Get the community of the given node."""
    node_u = G.nodes[u]
    try:
        return node_u[community]
    except KeyError as err:
        raise nx.NetworkXAlgorithmError('No community information') from err