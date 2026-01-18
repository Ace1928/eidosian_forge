import collections
import networkx as nx
def _restore_Tinout(popped_node1, popped_node2, graph_params, state_params):
    """Restores the previous version of Ti/Ti_out when a node pair is deleted from the mapping.

    Parameters
    ----------
    popped_node1, popped_node2: Graph node
        The two nodes deleted from the mapping.

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

        T1_tilde, T2_tilde: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti
    """
    G1, G2, _, _, _, _, _ = graph_params
    mapping, reverse_mapping, T1, T1_in, T1_tilde, T1_tilde_in, T2, T2_in, T2_tilde, T2_tilde_in = state_params
    is_added = False
    for neighbor in G1[popped_node1]:
        if neighbor in mapping:
            is_added = True
            T1.add(popped_node1)
        else:
            if any((nbr in mapping for nbr in G1[neighbor])):
                continue
            T1.discard(neighbor)
            T1_tilde.add(neighbor)
    if not is_added:
        T1_tilde.add(popped_node1)
    is_added = False
    for neighbor in G2[popped_node2]:
        if neighbor in reverse_mapping:
            is_added = True
            T2.add(popped_node2)
        else:
            if any((nbr in reverse_mapping for nbr in G2[neighbor])):
                continue
            T2.discard(neighbor)
            T2_tilde.add(neighbor)
    if not is_added:
        T2_tilde.add(popped_node2)