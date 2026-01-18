import copy
from collections import defaultdict
from itertools import combinations
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import (
from .utils import build_auxiliary_node_connectivity
def _is_separating_set(G, cut):
    """Assumes that the input graph is connected"""
    if len(cut) == len(G) - 1:
        return True
    H = nx.restricted_view(G, cut, [])
    if nx.is_connected(H):
        return False
    return True