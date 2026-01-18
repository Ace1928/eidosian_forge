from collections import Counter, defaultdict
from itertools import combinations, product
from math import inf
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
class _NeighborhoodCache(dict):
    """Very lightweight graph wrapper which caches neighborhoods as list.

    This dict subclass uses the __missing__ functionality to query graphs for
    their neighborhoods, and store the result as a list.  This is used to avoid
    the performance penalty incurred by subgraph views.
    """

    def __init__(self, G):
        self.G = G

    def __missing__(self, v):
        Gv = self[v] = list(self.G[v])
        return Gv