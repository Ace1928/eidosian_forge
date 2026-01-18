from sympy.core.backend import Matrix, eye, zeros
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import flatten
from sympy.physics.vector import dynamicsymbols
from sympy.physics.mechanics.functions import msubs
from collections import namedtuple
from collections.abc import Iterable
def _form_permutation_matrices(self):
    """Form the permutation matrices Pq and Pu."""
    l, m, n, o, s, k = self._dims
    if n != 0:
        self._Pq = permutation_matrix(self.q, Matrix([self.q_i, self.q_d]))
        if l > 0:
            self._Pqi = self._Pq[:, :-l]
            self._Pqd = self._Pq[:, -l:]
        else:
            self._Pqi = self._Pq
            self._Pqd = Matrix()
    if o != 0:
        self._Pu = permutation_matrix(self.u, Matrix([self.u_i, self.u_d]))
        if m > 0:
            self._Pui = self._Pu[:, :-m]
            self._Pud = self._Pu[:, -m:]
        else:
            self._Pui = self._Pu
            self._Pud = Matrix()
    P_col1 = Matrix([self._Pqi, zeros(o + k, n - l)])
    P_col2 = Matrix([zeros(n, o - m), self._Pui, zeros(k, o - m)])
    if P_col1:
        if P_col2:
            self.perm_mat = P_col1.row_join(P_col2)
        else:
            self.perm_mat = P_col1
    else:
        self.perm_mat = P_col2