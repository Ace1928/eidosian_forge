import itertools
import math
from collections import defaultdict
import networkx as nx
from networkx.utils import py_random_state
from .classic import complete_graph, empty_graph, path_graph, star_graph
from .degree_seq import degree_sequence_tree
def _suitable(edges, potential_edges):
    if not potential_edges:
        return True
    for s1 in potential_edges:
        for s2 in potential_edges:
            if s1 == s2:
                break
            if s1 > s2:
                s1, s2 = (s2, s1)
            if (s1, s2) not in edges:
                return True
    return False