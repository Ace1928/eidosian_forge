import re
import os
import numpy as np
import ase
from .vasp import Vasp
from ase.calculators.singlepoint import SinglePointCalculator
def _get_integrated_dos(self):
    if self._total_dos.shape[0] == 3:
        return self._total_dos[2, :]
    elif self._total_dos.shape[0] == 5:
        return self._total_dos[3:5, :]