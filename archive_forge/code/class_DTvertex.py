from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
class DTvertex(tuple):
    """
    A vertex of the 4-valent graph which is described by a DT code.
    Instantiate with an even-odd pair, in either order.
    """

    def __new__(self, pair, overcrossing=1):
        even_over = bool(overcrossing == -1)
        return tuple.__new__(self, (min(pair), max(pair), even_over))

    def __repr__(self):
        return str((self[0], self[1]))

    def entry_slot(self, N):
        if N == self[0]:
            return South
        elif N == self[1]:
            return East
        else:
            raise ValueError('%d is not a label of %s' % (N, self))

    def exit_slot(self, N):
        if N == self[0]:
            return North
        elif N == self[1]:
            return West
        else:
            raise ValueError('%d is not a label of %s' % (N, self))

    def upper_pair(self):
        first, second, even_over = self
        return (0, 2) if bool(first % 2) ^ even_over else (1, 3)