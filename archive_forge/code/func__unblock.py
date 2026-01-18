from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def _unblock(thisnode):
    """Recursively unblock and remove nodes from B[thisnode]."""
    if blocked[thisnode]:
        blocked[thisnode] = False
        while B[thisnode]:
            _unblock(B[thisnode].pop())