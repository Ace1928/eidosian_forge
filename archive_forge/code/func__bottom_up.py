from __future__ import annotations
from collections import deque
from dask.core import istask, subs
def _bottom_up(net, term):
    if istask(term):
        term = (head(term),) + tuple((_bottom_up(net, t) for t in args(term)))
    elif isinstance(term, list):
        term = [_bottom_up(net, t) for t in args(term)]
    return net._rewrite(term)