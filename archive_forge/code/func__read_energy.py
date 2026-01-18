import collections
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.units import Bohr, Hartree
from ase.utils import reader, writer
def _read_energy(self):
    txt = (self.path / 'TOTENERGY.OUT').read_text()
    tokens = txt.split()
    energy = float(tokens[-1]) * Hartree
    yield ('free_energy', energy)
    yield ('energy', energy)