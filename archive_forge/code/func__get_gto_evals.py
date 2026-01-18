import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def _get_gto_evals(chunk):
    spin = 1 if re.match('[ \\t\\S]+Beta', chunk) else 0
    data = []
    for vector in _extract_vector.finditer(chunk):
        data.append([float(x.replace('D', 'E')) for x in vector.groups()[1:]])
    data = np.array(data)
    occ = data[:, 0]
    energies = data[:, 1] * Hartree
    return SinglePointKPoint(1.0, spin, 0, energies, occ)