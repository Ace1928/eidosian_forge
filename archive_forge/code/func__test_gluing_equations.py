from ..sage_helper import _within_sage, doctest_modules
from ..pari import pari
import snappy
import snappy.snap as snap
import getopt
import sys
def _test_gluing_equations(manifold, shapes):
    """
    Given a manifold and exact shapes, test whether the rectangular gluing
    equations are fulfilled.
    """
    one_minus_shapes = [1 - shape for shape in shapes]
    for A, B, c in manifold.gluing_equations('rect'):
        val = c
        for a, shape in zip(A, shapes):
            val *= shape ** a
        for b, one_minus_shape in zip(B, one_minus_shapes):
            val *= one_minus_shape ** b
        if not val == 1:
            return False
    return True