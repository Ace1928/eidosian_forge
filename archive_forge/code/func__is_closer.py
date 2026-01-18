from collections import deque
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow.utils import build_residual_network
def _is_closer(u, v):
    return timestamp[v] <= timestamp[u] and dist[v] > dist[u] + 1