import networkx as nx
from networkx.utils.decorators import py_random_state
def _two_sweep_undirected(G, seed):
    """Helper function for finding a lower bound on the diameter
        for undirected Graphs.

        The idea is to pick the farthest node from a random node
        and return its eccentricity.

        ``G`` is a NetworkX undirected graph.

    .. note::

        ``seed`` is a random.Random or numpy.random.RandomState instance
    """
    source = seed.choice(list(G))
    distances = nx.shortest_path_length(G, source)
    if len(distances) != len(G):
        raise nx.NetworkXError('Graph not connected.')
    *_, node = distances
    return nx.eccentricity(G, node)