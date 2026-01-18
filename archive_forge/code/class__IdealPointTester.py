from ...sage_helper import _within_sage
from .finite_point import *
from .extended_matrix import *
class _IdealPointTester:
    """
    A test rig for idealPoint

    Run the test rig::

        sage: _IdealPointTester().run_tests()

    """

    def matrices(self):
        from sage.all import RIF, CIF, matrix
        return [matrix.identity(CIF, 2), matrix([[CIF(RIF(1.3), RIF(-0.4)), CIF(RIF(5.6), RIF(2.3))], [CIF(RIF(-0.3), RIF(0.1)), CIF(1)]]), matrix([[CIF(RIF(0.3), RIF(-1.4)), CIF(RIF(3.6), RIF(6.3))], [CIF(RIF(-0.3), RIF(1.1)), CIF(1)]])]

    def run_tests(self):
        from sage.all import RIF, CIF
        bias = RIF(1.5)
        triangle = [CIF(0), Infinity, CIF(1)]
        p = FinitePoint(CIF(0), 1 / bias)
        for m in self.matrices():
            pt = compute_midpoint_of_triangle_edge_with_offset([apply_Moebius(m, t) for t in triangle], bias)
            d = p.translate_PGL(m).dist(pt)
            if not d < RIF(1e-06):
                raise Exception('Points differ %s' % d)