from sympy.physics.mechanics import (Body, Lagrangian, KanesMethod, LagrangesMethod,
from sympy.physics.mechanics.method import _Methods
from sympy.core.backend import Matrix
def _generate_kdes(self):
    kd_ind = Matrix(1, 0, []).T
    for joint in self._joints:
        kd_ind = kd_ind.col_join(joint.kdes)
    return kd_ind