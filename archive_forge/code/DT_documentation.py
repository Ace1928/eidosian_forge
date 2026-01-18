from .. import FatGraph, FatEdge, Link, Crossing
from ..links.links import CrossingEntryPoint
from ..links.ordered_set import OrderedSet
from .Base64LikeDT import (decode_base64_like_DT_code, encode_base64_like_DT_code)

        Constructs a python simulation of a SnapPea KLPProjection
        (Kernel Link Projection) structure.  See DTFatGraph.KLP_dict
        and Jeff Weeks' SnapPea file link_projection.h for
        definitions.  Here the KLPCrossings are modeled by
        dictionaries.
        