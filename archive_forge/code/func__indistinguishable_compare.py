import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _indistinguishable_compare(self, atoms1, atoms2):
    """Finds each atom in atoms1's nearest neighbor with the same
        chemical symbol in atoms2. Return dmax, the farthest distance an
        individual atom differs by."""
    atoms2 = atoms2.copy()
    atoms2.set_constraint()
    dmax = 0.0
    for atom1 in atoms1:
        closest = [np.nan, np.inf]
        for index, atom2 in enumerate(atoms2):
            if atom2.symbol == atom1.symbol:
                d = np.linalg.norm(atom1.position - atom2.position)
                if d < closest[1]:
                    closest = [index, d]
        if closest[1] > dmax:
            dmax = closest[1]
        del atoms2[closest[0]]
    return dmax