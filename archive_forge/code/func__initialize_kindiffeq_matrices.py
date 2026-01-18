from sympy.core.backend import zeros, Matrix, diff, eye
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
def _initialize_kindiffeq_matrices(self, kdeqs):
    """Initialize the kinematic differential equation matrices.

        Parameters
        ==========
        kdeqs : sequence of sympy expressions
            Kinematic differential equations in the form of f(u,q',q,t) where
            f() = 0. The equations have to be linear in the generalized
            coordinates and generalized speeds.

        """
    if kdeqs:
        if len(self.q) != len(kdeqs):
            raise ValueError('There must be an equal number of kinematic differential equations and coordinates.')
        u = self.u
        qdot = self._qdot
        kdeqs = Matrix(kdeqs)
        u_zero = {ui: 0 for ui in u}
        uaux_zero = {uai: 0 for uai in self._uaux}
        qdot_zero = {qdi: 0 for qdi in qdot}
        k_ku = kdeqs.jacobian(u)
        k_kqdot = kdeqs.jacobian(qdot)
        f_k = kdeqs.xreplace(u_zero).xreplace(qdot_zero)
        dy_syms = find_dynamicsymbols(k_ku.row_join(k_kqdot).row_join(f_k))
        nonlin_vars = [vari for vari in u[:] + qdot[:] if vari in dy_syms]
        if nonlin_vars:
            msg = 'The provided kinematic differential equations are nonlinear in {}. They must be linear in the generalized speeds and derivatives of the generalized coordinates.'
            raise ValueError(msg.format(nonlin_vars))
        self._f_k_implicit = f_k.xreplace(uaux_zero)
        self._k_ku_implicit = k_ku.xreplace(uaux_zero)
        self._k_kqdot_implicit = k_kqdot
        f_k_explicit = k_kqdot.LUsolve(f_k)
        k_ku_explicit = k_kqdot.LUsolve(k_ku)
        self._qdot_u_map = dict(zip(qdot, -(k_ku_explicit * u + f_k_explicit)))
        self._f_k = f_k_explicit.xreplace(uaux_zero)
        self._k_ku = k_ku_explicit.xreplace(uaux_zero)
        self._k_kqdot = eye(len(qdot))
    else:
        self._qdot_u_map = None
        self._f_k_implicit = self._f_k = Matrix()
        self._k_ku_implicit = self._k_ku = Matrix()
        self._k_kqdot_implicit = self._k_kqdot = Matrix()