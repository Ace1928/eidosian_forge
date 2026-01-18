import random
import sys
from . import Nodes
def _add_nodedata(self, nd, st):
    """Add data to the node parsed from the comments, taxon and support (PRIVATE)."""
    if isinstance(st[1][-1], str) and st[1][-1].startswith(NODECOMMENT_START):
        nd.comment = st[1].pop(-1)
    elif isinstance(st[1][0], str):
        nd.taxon = st[1][0]
        st[1] = st[1][1:]
    if len(st) > 1:
        if len(st[1]) >= 2:
            nd.support = st[1][0]
            if st[1][1] is not None:
                nd.branchlength = st[1][1]
        elif len(st[1]) == 1:
            if not self.__values_are_support:
                if st[1][0] is not None:
                    nd.branchlength = st[1][0]
            else:
                nd.support = st[1][0]
    return nd