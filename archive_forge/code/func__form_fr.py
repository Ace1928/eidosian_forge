from sympy.core.backend import zeros, Matrix, diff, eye
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
def _form_fr(self, fl):
    """Form the generalized active force."""
    if fl is not None and (len(fl) == 0 or not iterable(fl)):
        raise ValueError('Force pairs must be supplied in an non-empty iterable or None.')
    N = self._inertial
    vel_list, f_list = _f_list_parser(fl, N)
    vel_list = [msubs(i, self._qdot_u_map) for i in vel_list]
    f_list = [msubs(i, self._qdot_u_map) for i in f_list]
    o = len(self.u)
    b = len(f_list)
    FR = zeros(o, 1)
    partials = partial_velocity(vel_list, self.u, N)
    for i in range(o):
        FR[i] = sum((partials[j][i] & f_list[j] for j in range(b)))
    if self._udep:
        p = o - len(self._udep)
        FRtilde = FR[:p, 0]
        FRold = FR[p:o, 0]
        FRtilde += self._Ars.T * FRold
        FR = FRtilde
    self._forcelist = fl
    self._fr = FR
    return FR