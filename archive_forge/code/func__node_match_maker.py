import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@staticmethod
def _node_match_maker(cmp):

    @wraps(cmp)
    def comparer(graph1, node1, graph2, node2):
        return cmp(graph1.nodes[node1], graph2.nodes[node2])
    return comparer