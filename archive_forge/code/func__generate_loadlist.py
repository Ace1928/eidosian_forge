from sympy.physics.mechanics import (Body, Lagrangian, KanesMethod, LagrangesMethod,
from sympy.physics.mechanics.method import _Methods
from sympy.core.backend import Matrix
def _generate_loadlist(self):
    load_list = []
    for body in self.bodies:
        load_list.extend(body.loads)
    return load_list