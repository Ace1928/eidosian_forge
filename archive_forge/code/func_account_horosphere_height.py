from ...sage_helper import _within_sage
from ...math_basics import correct_max
from ...snap.kernel_structures import *
from ...snap.fundamental_polyhedron import *
from ...snap.mcomplex_base import *
from ...snap.t3mlite import simplex
from ...snap import t3mlite as t3m
from ...exceptions import InsufficientPrecisionError
from ..cuspCrossSection import ComplexCuspCrossSection
from ..upper_halfspace.ideal_point import *
from ..interval_tree import *
from .cusp_translate_engine import *
import heapq
def account_horosphere_height(self, tile, vertex):
    horosphere_height = tile.height_of_horosphere(vertex, is_at_infinity=False)
    cusp = vertex.SubsimplexIndexInManifold
    self.max_horosphere_height_for_cusp[cusp] = correct_max([self.max_horosphere_height_for_cusp[cusp], horosphere_height])