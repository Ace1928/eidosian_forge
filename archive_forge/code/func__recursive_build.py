import itertools as it
from functools import partial
import networkx as nx
from networkx.utils import arbitrary_element, not_implemented_for
def _recursive_build(H, A, source, avail):
    if {source} == avail:
        return
    sink = arbitrary_element(avail - {source})
    value, (S, T) = nx.minimum_cut(H, source, sink)
    if H.is_directed():
        value_, (T_, S_) = nx.minimum_cut(H, sink, source)
        if value_ < value:
            value, S, T = (value_, S_, T_)
    A.add_edge(source, sink, weight=value)
    _recursive_build(H, A, source, avail.intersection(S))
    _recursive_build(H, A, sink, avail.intersection(T))