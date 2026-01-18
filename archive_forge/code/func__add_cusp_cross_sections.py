from ..sage_helper import _within_sage
import math
from ..snap import t3mlite as t3m
from ..snap.kernel_structures import *
from ..snap.mcomplex_base import *
from ..math_basics import correct_min
from .exceptions import *
def _add_cusp_cross_sections(self, one_cocycle):
    for T in self.mcomplex.Tetrahedra:
        T.horotriangles = {t3m.simplex.V0: None, t3m.simplex.V1: None, t3m.simplex.V2: None, t3m.simplex.V3: None}
    for cusp in self.mcomplex.Vertices:
        self._add_one_cusp_cross_section(cusp, one_cocycle)