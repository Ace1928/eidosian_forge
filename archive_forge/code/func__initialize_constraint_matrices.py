from sympy.core.backend import zeros, Matrix, diff, eye
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
def _initialize_constraint_matrices(self, config, vel, acc):
    """Initializes constraint matrices."""
    o = len(self.u)
    m = len(self._udep)
    p = o - m
    none_handler = lambda x: Matrix(x) if x else Matrix()
    config = none_handler(config)
    if len(self._qdep) != len(config):
        raise ValueError('There must be an equal number of dependent coordinates and configuration constraints.')
    self._f_h = none_handler(config)
    vel = none_handler(vel)
    acc = none_handler(acc)
    if len(vel) != m:
        raise ValueError('There must be an equal number of dependent speeds and velocity constraints.')
    if acc and len(acc) != m:
        raise ValueError('There must be an equal number of dependent speeds and acceleration constraints.')
    if vel:
        u_zero = {i: 0 for i in self.u}
        udot_zero = {i: 0 for i in self._udot}
        if self._qdot_u_map is not None:
            vel = msubs(vel, self._qdot_u_map)
        self._f_nh = msubs(vel, u_zero)
        self._k_nh = (vel - self._f_nh).jacobian(self.u)
        if not acc:
            _f_dnh = self._k_nh.diff(dynamicsymbols._t) * self.u + self._f_nh.diff(dynamicsymbols._t)
            if self._qdot_u_map is not None:
                _f_dnh = msubs(_f_dnh, self._qdot_u_map)
            self._f_dnh = _f_dnh
            self._k_dnh = self._k_nh
        else:
            if self._qdot_u_map is not None:
                acc = msubs(acc, self._qdot_u_map)
            self._f_dnh = msubs(acc, udot_zero)
            self._k_dnh = (acc - self._f_dnh).jacobian(self._udot)
        B_ind = self._k_nh[:, :p]
        B_dep = self._k_nh[:, p:o]
        self._Ars = -B_dep.LUsolve(B_ind)
    else:
        self._f_nh = Matrix()
        self._k_nh = Matrix()
        self._f_dnh = Matrix()
        self._k_dnh = Matrix()
        self._Ars = Matrix()