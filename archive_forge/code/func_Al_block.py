import pytest
from ase.build import bulk
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.contour_exploration import ContourExploration
import numpy as np
from ase.calculators.emt import EMT
from .test_ce_curvature import Al_atom_pair
def Al_block():
    size = 2
    atoms = bulk('Al', 'fcc', cubic=True).repeat((size, size, size))
    atoms.calc = EMT()
    return atoms