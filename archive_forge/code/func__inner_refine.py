from sympy.polys.densearith import (
from sympy.polys.densebasic import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.factortools import (
from sympy.polys.polyerrors import (
from sympy.polys.sqfreetools import (
def _inner_refine(self):
    """Internal one step complex root refinement procedure. """
    (u, v), (s, t) = (self.a, self.b)
    I, Q = (self.I, self.Q)
    f1, F1 = (self.f1, self.F1)
    f2, F2 = (self.f2, self.F2)
    dom = self.dom
    if s - u > t - v:
        D_L, D_R = _vertical_bisection(1, (u, v), (s, t), I, Q, F1, F2, f1, f2, dom)
        if D_L[0] == 1:
            _, a, b, I, Q, F1, F2 = D_L
        else:
            _, a, b, I, Q, F1, F2 = D_R
    else:
        D_B, D_U = _horizontal_bisection(1, (u, v), (s, t), I, Q, F1, F2, f1, f2, dom)
        if D_B[0] == 1:
            _, a, b, I, Q, F1, F2 = D_B
        else:
            _, a, b, I, Q, F1, F2 = D_U
    return ComplexInterval(a, b, I, Q, F1, F2, f1, f2, dom, self.conj)