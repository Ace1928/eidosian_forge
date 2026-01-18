import os
import re
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.dmol import write_dmol_car, write_dmol_incoor
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
 Cleanup after dmol calculation

        Only removes dmol files in self.directory,
        does not remove the directory itself
        