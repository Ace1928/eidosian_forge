import re
from collections import OrderedDict
import numpy as np
from ase import Atoms
from ase.units import Hartree, Bohr
from ase.calculators.singlepoint import (SinglePointDFTCalculator,
from .parser import _define_pattern
def _get_multipole(chunk):
    matches = _multipole.findall(chunk)
    if not matches:
        return (None, None)
    moments = [float(x.split()[4]) for x in matches[-1].split('\n') if x]
    dipole = np.array(moments[1:4]) * Bohr
    quadrupole = np.zeros(9)
    quadrupole[[0, 1, 2, 4, 5, 8]] = [moments[4:]]
    quadrupole[[3, 6, 7]] = quadrupole[[1, 2, 5]]
    return (dipole, quadrupole.reshape((3, 3)) * Bohr ** 2)