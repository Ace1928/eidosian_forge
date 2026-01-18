import collections
import networkx as nx
def _cut_PT(u, v, graph_params, state_params):
    """Implements the cutting rules for the ISO problem.

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

        T1_tilde, T2_tilde: set
            Ti_out contains all the nodes from Gi, that are neither in the mapping nor in Ti

    Returns
    -------
    True if we should prune this branch, i.e. the node pair failed the cutting checks. False otherwise.
    """
    G1, G2, G1_labels, G2_labels, _, _, _ = graph_params
    _, _, T1, T1_in, T1_tilde, _, T2, T2_in, T2_tilde, _ = state_params
    u_labels_predecessors, v_labels_predecessors = ({}, {})
    if G1.is_directed():
        u_labels_predecessors = nx.utils.groups({n1: G1_labels[n1] for n1 in G1.pred[u]})
        v_labels_predecessors = nx.utils.groups({n2: G2_labels[n2] for n2 in G2.pred[v]})
        if set(u_labels_predecessors.keys()) != set(v_labels_predecessors.keys()):
            return True
    u_labels_successors = nx.utils.groups({n1: G1_labels[n1] for n1 in G1[u]})
    v_labels_successors = nx.utils.groups({n2: G2_labels[n2] for n2 in G2[v]})
    if set(u_labels_successors.keys()) != set(v_labels_successors.keys()):
        return True
    for label, G1_nbh in u_labels_successors.items():
        G2_nbh = v_labels_successors[label]
        if G1.is_multigraph():
            u_nbrs_edges = sorted((G1.number_of_edges(u, x) for x in G1_nbh))
            v_nbrs_edges = sorted((G2.number_of_edges(v, x) for x in G2_nbh))
            if any((u_nbr_edges != v_nbr_edges for u_nbr_edges, v_nbr_edges in zip(u_nbrs_edges, v_nbrs_edges))):
                return True
        if len(T1.intersection(G1_nbh)) != len(T2.intersection(G2_nbh)):
            return True
        if len(T1_tilde.intersection(G1_nbh)) != len(T2_tilde.intersection(G2_nbh)):
            return True
        if G1.is_directed() and len(T1_in.intersection(G1_nbh)) != len(T2_in.intersection(G2_nbh)):
            return True
    if not G1.is_directed():
        return False
    for label, G1_pred in u_labels_predecessors.items():
        G2_pred = v_labels_predecessors[label]
        if G1.is_multigraph():
            u_pred_edges = sorted((G1.number_of_edges(u, x) for x in G1_pred))
            v_pred_edges = sorted((G2.number_of_edges(v, x) for x in G2_pred))
            if any((u_nbr_edges != v_nbr_edges for u_nbr_edges, v_nbr_edges in zip(u_pred_edges, v_pred_edges))):
                return True
        if len(T1.intersection(G1_pred)) != len(T2.intersection(G2_pred)):
            return True
        if len(T1_tilde.intersection(G1_pred)) != len(T2_tilde.intersection(G2_pred)):
            return True
        if len(T1_in.intersection(G1_pred)) != len(T2_in.intersection(G2_pred)):
            return True
    return False