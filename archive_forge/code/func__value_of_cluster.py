from copy import deepcopy
from functools import lru_cache
from random import choice
import networkx as nx
from networkx.utils import not_implemented_for
@lru_cache(CLUSTER_EVAL_CACHE_SIZE)
def _value_of_cluster(cluster):
    valid_edges = [e for e in safe_G.edges if e[0] in cluster and e[1] in cluster]
    return sum((safe_G.edges[e][edge_weight] for e in valid_edges))