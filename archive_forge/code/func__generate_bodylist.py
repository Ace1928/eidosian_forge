from sympy.physics.mechanics import (Body, Lagrangian, KanesMethod, LagrangesMethod,
from sympy.physics.mechanics.method import _Methods
from sympy.core.backend import Matrix
def _generate_bodylist(self):
    bodies = []
    for joint in self._joints:
        if joint.child not in bodies:
            bodies.append(joint.child)
        if joint.parent not in bodies:
            bodies.append(joint.parent)
    return bodies