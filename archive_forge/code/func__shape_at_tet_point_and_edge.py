from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def _shape_at_tet_point_and_edge(self, tet, pt, edge):
    """
        Given the index of a tetrahedron and two quadruples (any iterabel) of
        integers, give the cross ratio at that integral point and edge of that
        tetrahedron.
        This method translates the SnapPy conventions of labeling simplices
        and the conventions in Definition 4.2 of

        Garoufalidis, Goerner, Zickert:
        Gluing Equations for PGL(n,C)-Representations of 3-Manifolds
        http://arxiv.org/abs/1207.6711
        """
    postfix = '_%d%d%d%d' % tuple(pt) + '_%d' % tet
    if tuple(edge) in [(1, 1, 0, 0), (0, 0, 1, 1)]:
        return self['z' + postfix]
    if tuple(edge) in [(1, 0, 1, 0), (0, 1, 0, 1)]:
        return self['zp' + postfix]
    if tuple(edge) in [(1, 0, 0, 1), (0, 1, 1, 0)]:
        return self['zpp' + postfix]
    raise Exception('Invalid edge ' + str(edge))