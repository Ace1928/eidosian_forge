import pytest
import numpy as np
from ase import Atom
from ase.build import bulk
import ase.io
from ase import units
from ase.md.verlet import VelocityVerlet
@pytest.fixture
def Atoms_fcc_Ni_with_H_at_center():
    atoms = bulk('Ni', cubic=True)
    atoms += Atom('H', position=atoms.cell.diagonal() / 2)
    return atoms