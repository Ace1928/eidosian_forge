from collections import defaultdict
import networkx as nx
from networkx.algorithms.community.quality import modularity
from networkx.utils import not_implemented_for
from networkx.utils.mapped_queue import MappedQueue
def _greedy_modularity_communities_generator(G, weight=None, resolution=1):
    """Yield community partitions of G and the modularity change at each step.

    This function performs Clauset-Newman-Moore greedy modularity maximization [2]_
    At each step of the process it yields the change in modularity that will occur in
    the next step followed by yielding the new community partition after that step.

    Greedy modularity maximization begins with each node in its own community
    and repeatedly joins the pair of communities that lead to the largest
    modularity until one community contains all nodes (the partition has one set).

    This function maximizes the generalized modularity, where `resolution`
    is the resolution parameter, often expressed as $\\gamma$.
    See :func:`~networkx.algorithms.community.quality.modularity`.

    Parameters
    ----------
    G : NetworkX graph

    weight : string or None, optional (default=None)
        The name of an edge attribute that holds the numerical value used
        as a weight.  If None, then each edge has weight 1.
        The degree is the sum of the edge weights adjacent to the node.

    resolution : float (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    Yields
    ------
    Alternating yield statements produce the following two objects:

    communities: dict_values
        A dict_values of frozensets of nodes, one for each community.
        This represents a partition of the nodes of the graph into communities.
        The first yield is the partition with each node in its own community.

    dq: float
        The change in modularity when merging the next two communities
        that leads to the largest modularity.

    See Also
    --------
    modularity

    References
    ----------
    .. [1] Newman, M. E. J. "Networks: An Introduction", page 224
       Oxford University Press 2011.
    .. [2] Clauset, A., Newman, M. E., & Moore, C.
       "Finding community structure in very large networks."
       Physical Review E 70(6), 2004.
    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community
       Detection" Phys. Rev. E74, 2006.
    .. [4] Newman, M. E. J."Analysis of weighted networks"
       Physical Review E 70(5 Pt 2):056131, 2004.
    """
    directed = G.is_directed()
    N = G.number_of_nodes()
    m = G.size(weight)
    q0 = 1 / m
    if directed:
        a = {node: deg_out * q0 for node, deg_out in G.out_degree(weight=weight)}
        b = {node: deg_in * q0 for node, deg_in in G.in_degree(weight=weight)}
    else:
        a = b = {node: deg * q0 * 0.5 for node, deg in G.degree(weight=weight)}
    dq_dict = defaultdict(lambda: defaultdict(float))
    for u, v, wt in G.edges(data=weight, default=1):
        if u == v:
            continue
        dq_dict[u][v] += wt
        dq_dict[v][u] += wt
    for u, nbrdict in dq_dict.items():
        for v, wt in nbrdict.items():
            dq_dict[u][v] = q0 * wt - resolution * (a[u] * b[v] + b[u] * a[v])
    dq_heap = {u: MappedQueue({(u, v): -dq for v, dq in dq_dict[u].items()}) for u in G}
    H = MappedQueue([dq_heap[n].heap[0] for n in G if len(dq_heap[n]) > 0])
    communities = {n: frozenset([n]) for n in G}
    yield communities.values()
    while len(H) > 1:
        try:
            negdq, u, v = H.pop()
        except IndexError:
            break
        dq = -negdq
        yield dq
        dq_heap[u].pop()
        if len(dq_heap[u]) > 0:
            H.push(dq_heap[u].heap[0])
        if dq_heap[v].heap[0] == (v, u):
            H.remove((v, u))
            dq_heap[v].remove((v, u))
            if len(dq_heap[v]) > 0:
                H.push(dq_heap[v].heap[0])
        else:
            dq_heap[v].remove((v, u))
        communities[v] = frozenset(communities[u] | communities[v])
        del communities[u]
        u_nbrs = set(dq_dict[u])
        v_nbrs = set(dq_dict[v])
        all_nbrs = (u_nbrs | v_nbrs) - {u, v}
        both_nbrs = u_nbrs & v_nbrs
        for w in all_nbrs:
            if w in both_nbrs:
                dq_vw = dq_dict[v][w] + dq_dict[u][w]
            elif w in v_nbrs:
                dq_vw = dq_dict[v][w] - resolution * (a[u] * b[w] + a[w] * b[u])
            else:
                dq_vw = dq_dict[u][w] - resolution * (a[v] * b[w] + a[w] * b[v])
            for row, col in [(v, w), (w, v)]:
                dq_heap_row = dq_heap[row]
                dq_dict[row][col] = dq_vw
                if len(dq_heap_row) > 0:
                    d_oldmax = dq_heap_row.heap[0]
                else:
                    d_oldmax = None
                d = (row, col)
                d_negdq = -dq_vw
                if w in v_nbrs:
                    dq_heap_row.update(d, d, priority=d_negdq)
                else:
                    dq_heap_row.push(d, priority=d_negdq)
                if d_oldmax is None:
                    H.push(d, priority=d_negdq)
                else:
                    row_max = dq_heap_row.heap[0]
                    if d_oldmax != row_max or d_oldmax.priority != row_max.priority:
                        H.update(d_oldmax, row_max)
        for w in dq_dict[u]:
            dq_old = dq_dict[w][u]
            del dq_dict[w][u]
            if w != v:
                for row, col in [(w, u), (u, w)]:
                    dq_heap_row = dq_heap[row]
                    d_old = (row, col)
                    if dq_heap_row.heap[0] == d_old:
                        dq_heap_row.remove(d_old)
                        H.remove(d_old)
                        if len(dq_heap_row) > 0:
                            H.push(dq_heap_row.heap[0])
                    else:
                        dq_heap_row.remove(d_old)
        del dq_dict[u]
        dq_heap[u] = MappedQueue()
        a[v] += a[u]
        a[u] = 0
        if directed:
            b[v] += b[u]
            b[u] = 0
        yield communities.values()