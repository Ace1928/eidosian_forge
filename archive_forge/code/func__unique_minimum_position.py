import os
import numpy as np
from ase import io, units
from ase.optimize import QuasiNewton
from ase.parallel import paropen, world
from ase.md import VelocityVerlet
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
def _unique_minimum_position(self):
    """Identifies if the current position of the atoms, which should be
        a local minima, has been found before."""
    unique = True
    dmax_closest = 99999.0
    compare = ComparePositions(translate=True)
    self._read_minima()
    for minimum in self._minima:
        dmax = compare(minimum, self._atoms)
        if dmax < self._minima_threshold:
            unique = False
        if dmax < dmax_closest:
            dmax_closest = dmax
    return (unique, dmax_closest)