from sympy.core.function import expand_mul
from sympy.core.numbers import pi
from sympy.core.singleton import S
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.core.backend import Matrix, _simplify_matrix, eye, zeros
from sympy.core.symbol import symbols
from sympy.physics.mechanics import (dynamicsymbols, Body, JointsMethod,
from sympy.physics.mechanics.joint import Joint
from sympy.physics.vector import Vector, ReferenceFrame, Point
from sympy.testing.pytest import raises, warns_deprecated_sympy
def _generate_body(interframe=False):
    N = ReferenceFrame('N')
    A = ReferenceFrame('A')
    P = Body('P', frame=N)
    C = Body('C', frame=A)
    if interframe:
        Pint, Cint = (ReferenceFrame('P_int'), ReferenceFrame('C_int'))
        Pint.orient_axis(N, N.x, pi)
        Cint.orient_axis(A, A.y, -pi / 2)
        return (N, A, P, C, Pint, Cint)
    return (N, A, P, C)