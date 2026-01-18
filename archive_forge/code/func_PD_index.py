from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)
def PD_index(self):
    """
        The labelling of vertices when building a DT code also
        determines a labelling of the edges, which is needed
        for generating a PD description of the diagram.
        This method returns the edge label.
        """
    v = self[0]
    if self.slot(v) % 2 == 0:
        return v[0]
    else:
        return v[1]