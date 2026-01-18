from snappy import *
from snappy.SnapPy import triangulate_link_complement_from_data
from spherogram import FatGraph, FatEdge, CyclicList, Link, Crossing
import string

        Constructs a python simulation of a SnapPea KLPProjection
        (Kernel Link Projection) structure.  See DTFatGraph.KLP_dict
        and Jeff Weeks' SnapPea file link_projection.h for
        definitions.  Here the KLPCrossings are modeled by
        dictionaries.
        