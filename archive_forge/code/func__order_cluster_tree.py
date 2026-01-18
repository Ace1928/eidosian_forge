import warnings
import bisect
from collections import deque
import numpy as np
from . import _hierarchy, _optimal_leaf_ordering
import scipy.spatial.distance as distance
from scipy._lib._array_api import array_namespace, as_xparray, copy
from scipy._lib._disjoint_set import DisjointSet
def _order_cluster_tree(Z):
    """
    Return clustering nodes in bottom-up order by distance.

    Parameters
    ----------
    Z : scipy.cluster.linkage array
        The linkage matrix.

    Returns
    -------
    nodes : list
        A list of ClusterNode objects.
    """
    q = deque()
    tree = to_tree(Z)
    q.append(tree)
    nodes = []
    while q:
        node = q.popleft()
        if not node.is_leaf():
            bisect.insort_left(nodes, node)
            q.append(node.get_right())
            q.append(node.get_left())
    return nodes