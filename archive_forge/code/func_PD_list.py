from snappy import *
from snappy.SnapPy import triangulate_link_complement_from_data
from spherogram import FatGraph, FatEdge, CyclicList, Link, Crossing
import string
def PD_list(self, vertex):
    """
        Return the PD labels of the incident edges in order, starting
        with the incoming undercrossing as required for PD codes.
        """
    edgelist = [e.PD_index() for e in self(vertex)]
    n = edgelist.index(vertex.first_under())
    return edgelist[n:] + edgelist[:n]