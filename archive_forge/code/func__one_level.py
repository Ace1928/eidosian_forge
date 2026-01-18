from collections import defaultdict, deque
import networkx as nx
from networkx.algorithms.community import modularity
from networkx.utils import py_random_state
def _one_level(G, m, partition, resolution=1, is_directed=False, seed=None):
    """Calculate one level of the Louvain partitions tree

    Parameters
    ----------
    G : NetworkX Graph/DiGraph
        The graph from which to detect communities
    m : number
        The size of the graph `G`.
    partition : list of sets of nodes
        A valid partition of the graph `G`
    resolution : positive number
        The resolution parameter for computing the modularity of a partition
    is_directed : bool
        True if `G` is a directed graph.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    """
    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]
    if is_directed:
        in_degrees = dict(G.in_degree(weight='weight'))
        out_degrees = dict(G.out_degree(weight='weight'))
        Stot_in = list(in_degrees.values())
        Stot_out = list(out_degrees.values())
        nbrs = {}
        for u in G:
            nbrs[u] = defaultdict(float)
            for _, n, wt in G.out_edges(u, data='weight'):
                if u != n:
                    nbrs[u][n] += wt
            for n, _, wt in G.in_edges(u, data='weight'):
                if u != n:
                    nbrs[u][n] += wt
    else:
        degrees = dict(G.degree(weight='weight'))
        Stot = list(degrees.values())
        nbrs = {u: {v: data['weight'] for v, data in G[u].items() if v != u} for u in G}
    rand_nodes = list(G.nodes)
    seed.shuffle(rand_nodes)
    nb_moves = 1
    improvement = False
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:
            best_mod = 0
            best_com = node2com[u]
            weights2com = _neighbor_weights(nbrs[u], node2com)
            if is_directed:
                in_degree = in_degrees[u]
                out_degree = out_degrees[u]
                Stot_in[best_com] -= in_degree
                Stot_out[best_com] -= out_degree
                remove_cost = -weights2com[best_com] / m + resolution * (out_degree * Stot_in[best_com] + in_degree * Stot_out[best_com]) / m ** 2
            else:
                degree = degrees[u]
                Stot[best_com] -= degree
                remove_cost = -weights2com[best_com] / m + resolution * (Stot[best_com] * degree) / (2 * m ** 2)
            for nbr_com, wt in weights2com.items():
                if is_directed:
                    gain = remove_cost + wt / m - resolution * (out_degree * Stot_in[nbr_com] + in_degree * Stot_out[nbr_com]) / m ** 2
                else:
                    gain = remove_cost + wt / m - resolution * (Stot[nbr_com] * degree) / (2 * m ** 2)
                if gain > best_mod:
                    best_mod = gain
                    best_com = nbr_com
            if is_directed:
                Stot_in[best_com] += in_degree
                Stot_out[best_com] += out_degree
            else:
                Stot[best_com] += degree
            if best_com != node2com[u]:
                com = G.nodes[u].get('nodes', {u})
                partition[node2com[u]].difference_update(com)
                inner_partition[node2com[u]].remove(u)
                partition[best_com].update(com)
                inner_partition[best_com].add(u)
                improvement = True
                nb_moves += 1
                node2com[u] = best_com
    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    return (partition, inner_partition, improvement)