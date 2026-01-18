from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def KLP_strand(self, vertex, edge):
    """
        Return the SnapPea KLP strand name for the given edge at the
        end opposite to the vertex.
        """
    W = edge(vertex)
    slot = edge.slot(W)
    return 'X' if (slot == 0 or slot == 2) ^ self.flipped(W) else 'Y'