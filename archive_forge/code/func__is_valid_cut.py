import random
import networkx as nx
from networkx.algorithms.approximation import maxcut
def _is_valid_cut(G, set1, set2):
    union = set1.union(set2)
    assert union == set(G.nodes)
    assert len(set1) + len(set2) == G.number_of_nodes()