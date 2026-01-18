import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
def _unpack_available_edges(avail, weight=None, G=None):
    """Helper to separate avail into edges and corresponding weights"""
    if weight is None:
        weight = 'weight'
    if isinstance(avail, dict):
        avail_uv = list(avail.keys())
        avail_w = list(avail.values())
    else:

        def _try_getitem(d):
            try:
                return d[weight]
            except TypeError:
                return d
        avail_uv = [tup[0:2] for tup in avail]
        avail_w = [1 if len(tup) == 2 else _try_getitem(tup[-1]) for tup in avail]
    if G is not None:
        flags = [not G.has_edge(u, v) for u, v in avail_uv]
        avail_uv = list(it.compress(avail_uv, flags))
        avail_w = list(it.compress(avail_w, flags))
    return (avail_uv, avail_w)