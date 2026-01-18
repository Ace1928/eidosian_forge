import collections
import networkx as nx
def _update_Tinout(new_node1, new_node2, graph_params, state_params):
    """Updates the Ti/Ti_out (i=1,2) when a new node pair u-v is added to the mapping.

    Notes
    -----
    This function should be called right after the feasibility checks are passed, and node1 is mapped to node2. The
    purpose of this function is to avoid brute force computing of Ti/Ti_out by iterating over all nodes of the graph
    and checking which nodes satisfy the necessary conditions. Instead, in every step of the algorithm we focus
    exclusively on the two nodes that are being added to the mapping, incrementally updating Ti/Ti_out.

    Parameters
    ----------
    new_node1, new_node2: Graph node
        The two new nodes, added to the mapping.

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
    uncovered_successors_G1 = {succ for succ in G1[new_node1] if succ not in mapping}
    uncovered_successors_G2 = {succ for succ in G2[new_node2] if succ not in reverse_mapping}
    T1.update(uncovered_successors_G1)
    T2.update(uncovered_successors_G2)
    T1.discard(new_node1)
    T2.discard(new_node2)
    T1_tilde.difference_update(uncovered_successors_G1)
    T2_tilde.difference_update(uncovered_successors_G2)
    T1_tilde.discard(new_node1)
    T2_tilde.discard(new_node2)
    if not G1.is_directed():
        return
    uncovered_predecessors_G1 = {pred for pred in G1.pred[new_node1] if pred not in mapping}
    uncovered_predecessors_G2 = {pred for pred in G2.pred[new_node2] if pred not in reverse_mapping}
    T1_in.update(uncovered_predecessors_G1)
    T2_in.update(uncovered_predecessors_G2)
    T1_in.discard(new_node1)
    T2_in.discard(new_node2)
    T1_tilde.difference_update(uncovered_predecessors_G1)
    T2_tilde.difference_update(uncovered_predecessors_G2)
    T1_tilde.discard(new_node1)
    T2_tilde.discard(new_node2)