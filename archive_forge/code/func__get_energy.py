import re
import os
import numpy as np
import ase
from .vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
def _get_energy(self):
    """Return the array with the energies."""
    return self._total_dos[0, :]