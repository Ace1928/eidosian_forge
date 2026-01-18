from sympy.core.backend import Matrix, eye, zeros
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import flatten
from sympy.physics.vector import dynamicsymbols
from sympy.physics.mechanics.functions import msubs
from collections import namedtuple
from collections.abc import Iterable
def _form_coefficient_matrices(self):
    """Form the coefficient matrices C_0, C_1, and C_2."""
    l, m, n, o, s, k = self._dims
    if l > 0:
        f_c_jac_q = self.f_c.jacobian(self.q)
        self._C_0 = (eye(n) - self._Pqd * (f_c_jac_q * self._Pqd).LUsolve(f_c_jac_q)) * self._Pqi
    else:
        self._C_0 = eye(n)
    if m > 0:
        f_v_jac_u = self.f_v.jacobian(self.u)
        temp = f_v_jac_u * self._Pud
        if n != 0:
            f_v_jac_q = self.f_v.jacobian(self.q)
            self._C_1 = -self._Pud * temp.LUsolve(f_v_jac_q)
        else:
            self._C_1 = zeros(o, n)
        self._C_2 = (eye(o) - self._Pud * temp.LUsolve(f_v_jac_u)) * self._Pui
    else:
        self._C_1 = zeros(o, n)
        self._C_2 = eye(o)