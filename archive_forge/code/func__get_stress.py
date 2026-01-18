import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def _get_stress(chunk, cell):
    stress_blocks = _stress.findall(chunk)
    if not stress_blocks:
        return None
    stress_block = stress_blocks[-1]
    stress = np.zeros((3, 3))
    for i, row in enumerate(stress_block.strip().split('\n')):
        stress[i] = [float(x) for x in row.split()[1:4]]
    stress = stress @ cell * Hartree / Bohr / cell.volume
    stress = 0.5 * (stress + stress.T)
    return stress.ravel()[[0, 4, 8, 5, 2, 1]]