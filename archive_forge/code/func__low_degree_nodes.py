import itertools as it
from functools import partial
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for
def _low_degree_nodes(G, k, nbunch=None):
    """Helper for finding nodes with degree less than k."""
    if G.is_directed():
        seen = set()
        for node, degree in G.out_degree(nbunch):
            if degree < k:
                seen.add(node)
                yield node
        for node, degree in G.in_degree(nbunch):
            if node not in seen and degree < k:
                seen.add(node)
                yield node
    else:
        for node, degree in G.degree(nbunch):
            if degree < k:
                yield node