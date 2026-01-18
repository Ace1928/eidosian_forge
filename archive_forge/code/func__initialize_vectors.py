from sympy.core.backend import zeros, Matrix, diff, eye
from sympy.core.sorting import default_sort_key
from sympy.physics.vector import (ReferenceFrame, dynamicsymbols,
from sympy.physics.mechanics.method import _Methods
from sympy.physics.mechanics.particle import Particle
from sympy.physics.mechanics.rigidbody import RigidBody
from sympy.physics.mechanics.functions import (
from sympy.physics.mechanics.linearize import Linearizer
from sympy.utilities.iterables import iterable
def _initialize_vectors(self, q_ind, q_dep, u_ind, u_dep, u_aux):
    """Initialize the coordinate and speed vectors."""
    none_handler = lambda x: Matrix(x) if x else Matrix()
    q_dep = none_handler(q_dep)
    if not iterable(q_ind):
        raise TypeError('Generalized coordinates must be an iterable.')
    if not iterable(q_dep):
        raise TypeError('Dependent coordinates must be an iterable.')
    q_ind = Matrix(q_ind)
    self._qdep = q_dep
    self._q = Matrix([q_ind, q_dep])
    self._qdot = self.q.diff(dynamicsymbols._t)
    u_dep = none_handler(u_dep)
    if not iterable(u_ind):
        raise TypeError('Generalized speeds must be an iterable.')
    if not iterable(u_dep):
        raise TypeError('Dependent speeds must be an iterable.')
    u_ind = Matrix(u_ind)
    self._udep = u_dep
    self._u = Matrix([u_ind, u_dep])
    self._udot = self.u.diff(dynamicsymbols._t)
    self._uaux = none_handler(u_aux)