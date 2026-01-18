import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
def _core_subgraph(G, k_filter, k=None, core=None):
    """Returns the subgraph induced by nodes passing filter `k_filter`.

    Parameters
    ----------
    G : NetworkX graph
       The graph or directed graph to process
    k_filter : filter function
       This function filters the nodes chosen. It takes three inputs:
       A node of G, the filter's cutoff, and the core dict of the graph.
       The function should return a Boolean value.
    k : int, optional
      The order of the core. If not specified use the max core number.
      This value is used as the cutoff for the filter.
    core : dict, optional
      Precomputed core numbers keyed by node for the graph `G`.
      If not specified, the core numbers will be computed from `G`.

    """
    if core is None:
        core = core_number(G)
    if k is None:
        k = max(core.values())
    nodes = (v for v in core if k_filter(v, k, core))
    return G.subgraph(nodes).copy()