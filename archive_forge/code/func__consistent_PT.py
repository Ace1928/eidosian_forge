import collections
import networkx as nx
def _consistent_PT(u, v, graph_params, state_params):
    """Checks the consistency of extending the mapping using the current node pair.

    Parameters
    ----------
    u, v: Graph node
        The two candidate nodes being examined.

    graph_params: namedtuple
        Contains all the Graph-related parameters:

        G1,G2: NetworkX Graph or MultiGraph instances.
            The two graphs to check for isomorphism or monomorphism

        G1_labels,G2_labels: dict
            The label of every node in G1 and G2 respectively

    state_params: namedtuple
        Contains all the State-related parameters:

        mapping: dict
            The mapping as extended so far. Maps nodes of G1 to nodes of G2

        reverse_mapping: dict
            The reverse mapping as extended so far. Maps nodes from G2 to nodes of G1. It's basically "mapping" reversed

        T1, T2: set
            Ti contains uncovered neighbors of covered nodes from Gi, i.e. nodes that are not in the mapping, but are
            neighbors of nodes that are.

        T1_out, T2_out: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti

    Returns
    -------
    True if the pair passes all the consistency checks successfully. False otherwise.
    """
    G1, G2 = (graph_params.G1, graph_params.G2)
    mapping, reverse_mapping = (state_params.mapping, state_params.reverse_mapping)
    for neighbor in G1[u]:
        if neighbor in mapping:
            if G1.number_of_edges(u, neighbor) != G2.number_of_edges(v, mapping[neighbor]):
                return False
    for neighbor in G2[v]:
        if neighbor in reverse_mapping:
            if G1.number_of_edges(u, reverse_mapping[neighbor]) != G2.number_of_edges(v, neighbor):
                return False
    if not G1.is_directed():
        return True
    for predecessor in G1.pred[u]:
        if predecessor in mapping:
            if G1.number_of_edges(predecessor, u) != G2.number_of_edges(mapping[predecessor], v):
                return False
    for predecessor in G2.pred[v]:
        if predecessor in reverse_mapping:
            if G1.number_of_edges(reverse_mapping[predecessor], u) != G2.number_of_edges(predecessor, v):
                return False
    return True