from copy import deepcopy
from functools import lru_cache
from random import choice
import networkx as nx
from networkx.utils import not_implemented_for
def _pivot(partition, node):
    ccx = [c for c in partition if node in c]
    assert len(ccx) == 1
    return ccx[0]