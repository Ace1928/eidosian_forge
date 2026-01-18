from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def PD_tuple(self, vertex):
    """
        Return the PD labels of the incident edges in order, starting
        with the incoming undercrossing as required for PD codes.
        """
    edgelist = [e.PD_index() for e in self(vertex)]
    n = edgelist.index(self.incoming_under(vertex))
    return tuple(edgelist[n:] + edgelist[:n])