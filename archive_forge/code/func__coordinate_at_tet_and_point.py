from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _coordinate_at_tet_and_point(self, tet, pt):
    """
        Given the index of a tetrahedron and a quadruple (any iterable) of
        integer to mark an integral point on that tetrahedron, returns the
        associated Ptolemy coordinate.
        If this is a vertex Ptolemy coordinate, always return 1 without
        checking for it in the dictionary.
        """
    if sum(pt) in pt:
        return 1
    return self['c_%d%d%d%d' % tuple(pt) + '_%d' % tet]