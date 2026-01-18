from snappy import *
from snappy.SnapPy import triangulate_link_complement_from_data
from spherogram import FatGraph, FatEdge, CyclicList, Link, Crossing
import string
def PD(self, KnotTheory=False):
    G = self.fat_graph
    PD = [G.PD_list(v) for v in G.vertices]
    if KnotTheory:
        PD = 'PD' + repr(PD).replace('[', 'X[')[1:]
    return PD