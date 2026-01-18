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
class UngluedGenerator:

    def __init__(self, tile, g, height_upper_bound):
        self.tile = tile
        self.g = g
        self.height_upper_bound = height_upper_bound

    def _key(self):
        return (-self.height_upper_bound, self.g)

    def __lt__(self, other):
        return self._key() < other._key()