import networkx as nx
from networkx.utils import not_implemented_for
def _extrema_bounding(G, compute='diameter', weight=None):
    """Compute requested extreme distance metric of undirected graph G

    Computation is based on smart lower and upper bounds, and in practice
    linear in the number of nodes, rather than quadratic (except for some
    border cases such as complete graphs or circle shaped graphs).

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph

    compute : string denoting the requesting metric
       "diameter" for the maximal eccentricity value,
       "radius" for the minimal eccentricity value,
       "periphery" for the set of nodes with eccentricity equal to the diameter,
       "center" for the set of nodes with eccentricity equal to the radius,
       "eccentricities" for the maximum distance from each node to all other nodes in G

    weight : string, function, or None
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.

        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.

        If this is None, every edge has weight/distance/cost 1.

        Weights stored as floating point values can lead to small round-off
        errors in distances. Use integer weights to avoid this.

        Weights should be positive, since they are distances.

    Returns
    -------
    value : value of the requested metric
       int for "diameter" and "radius" or
       list of nodes for "center" and "periphery" or
       dictionary of eccentricity values keyed by node for "eccentricities"

    Raises
    ------
    NetworkXError
        If the graph consists of multiple components
    ValueError
        If `compute` is not one of "diameter", "radius", "periphery", "center", or "eccentricities".

    Notes
    -----
    This algorithm was proposed in [1]_ and discussed further in [2]_ and [3]_.

    References
    ----------
    .. [1] F. W. Takes, W. A. Kosters,
       "Determining the diameter of small world networks."
       Proceedings of the 20th ACM international conference on Information and knowledge management, 2011
       https://dl.acm.org/doi/abs/10.1145/2063576.2063748
    .. [2] F. W. Takes, W. A. Kosters,
       "Computing the Eccentricity Distribution of Large Graphs."
       Algorithms, 2013
       https://www.mdpi.com/1999-4893/6/1/100
    .. [3] M. Borassi, P. Crescenzi, M. Habib, W. A. Kosters, A. Marino, F. W. Takes,
       "Fast diameter and radius BFS-based computation in (weakly connected) real-world graphs: With an application to the six degrees of separation games. "
       Theoretical Computer Science, 2015
       https://www.sciencedirect.com/science/article/pii/S0304397515001644
    """
    degrees = dict(G.degree())
    minlowernode = max(degrees, key=degrees.get)
    N = len(degrees)
    high = False
    ecc_lower = dict.fromkeys(G, 0)
    ecc_upper = dict.fromkeys(G, N)
    candidates = set(G)
    minlower = N
    maxlower = 0
    minupper = N
    maxupper = 0
    while candidates:
        if high:
            current = maxuppernode
        else:
            current = minlowernode
        high = not high
        dist = nx.shortest_path_length(G, source=current, weight=weight)
        if len(dist) != N:
            msg = 'Cannot compute metric because graph is not connected.'
            raise nx.NetworkXError(msg)
        current_ecc = max(dist.values())
        maxuppernode = None
        minlowernode = None
        for i in candidates:
            d = dist[i]
            ecc_lower[i] = low = max(ecc_lower[i], max(d, current_ecc - d))
            ecc_upper[i] = upp = min(ecc_upper[i], current_ecc + d)
            minlower = min(ecc_lower[i], minlower)
            maxlower = max(ecc_lower[i], maxlower)
            minupper = min(ecc_upper[i], minupper)
            maxupper = max(ecc_upper[i], maxupper)
        if compute == 'diameter':
            ruled_out = {i for i in candidates if ecc_upper[i] <= maxlower and 2 * ecc_lower[i] >= maxupper}
        elif compute == 'radius':
            ruled_out = {i for i in candidates if ecc_lower[i] >= minupper and ecc_upper[i] + 1 <= 2 * minlower}
        elif compute == 'periphery':
            ruled_out = {i for i in candidates if ecc_upper[i] < maxlower and (maxlower == maxupper or ecc_lower[i] > maxupper)}
        elif compute == 'center':
            ruled_out = {i for i in candidates if ecc_lower[i] > minupper and (minlower == minupper or ecc_upper[i] + 1 < 2 * minlower)}
        elif compute == 'eccentricities':
            ruled_out = set()
        else:
            msg = "compute must be one of 'diameter', 'radius', 'periphery', 'center', 'eccentricities'"
            raise ValueError(msg)
        ruled_out.update((i for i in candidates if ecc_lower[i] == ecc_upper[i]))
        candidates -= ruled_out
        for i in candidates:
            if minlowernode is None or (ecc_lower[i] == ecc_lower[minlowernode] and degrees[i] > degrees[minlowernode]) or ecc_lower[i] < ecc_lower[minlowernode]:
                minlowernode = i
            if maxuppernode is None or (ecc_upper[i] == ecc_upper[maxuppernode] and degrees[i] > degrees[maxuppernode]) or ecc_upper[i] > ecc_upper[maxuppernode]:
                maxuppernode = i
    if compute == 'diameter':
        return maxlower
    if compute == 'radius':
        return minupper
    if compute == 'periphery':
        p = [v for v in G if ecc_lower[v] == maxlower]
        return p
    if compute == 'center':
        c = [v for v in G if ecc_upper[v] == minupper]
        return c
    if compute == 'eccentricities':
        return ecc_lower
    return None