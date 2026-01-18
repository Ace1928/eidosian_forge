from sympy.physics.mechanics import (Body, Lagrangian, KanesMethod, LagrangesMethod,
from sympy.physics.mechanics.method import _Methods
from sympy.core.backend import Matrix
def _generate_u(self):
    u_ind = []
    for joint in self._joints:
        for speed in joint.speeds:
            if speed in u_ind:
                raise ValueError('Speeds of joints should be unique.')
            u_ind.append(speed)
    return Matrix(u_ind)