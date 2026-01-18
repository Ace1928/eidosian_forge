from collections import deque
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
def _has_valid_root(n, tree):
    path = []
    v = n
    while v is not None:
        path.append(v)
        if v in (s, t):
            base_dist = 0
            break
        elif timestamp[v] == time:
            base_dist = dist[v]
            break
        v = tree[v]
    else:
        return False
    length = len(path)
    for i, u in enumerate(path, 1):
        dist[u] = base_dist + length - i
        timestamp[u] = time
    return True