from copy import deepcopy
from functools import lru_cache
from random import choice
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('undirected')
def _a_parent_of_leaves_only(gr):
    tleaves = set(_leaves(gr))
    for n in set(gr.nodes) - tleaves:
        if all((x in tleaves for x in nx.descendants(gr, n))):
            return n