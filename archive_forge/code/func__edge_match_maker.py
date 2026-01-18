import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@staticmethod
def _edge_match_maker(cmp):

    @wraps(cmp)
    def comparer(graph1, edge1, graph2, edge2):
        return cmp(graph1.edges[edge1], graph2.edges[edge2])
    return comparer