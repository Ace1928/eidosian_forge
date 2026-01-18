import os
import re
import tempfile
import warnings
import shutil
from os.path import join, isfile, islink
import numpy as np
from ase.units import Ry, eV, Bohr
from ase.data import atomic_numbers
from ase.io.siesta import read_siesta_xv
from ase.calculators.siesta.import_functions import read_rho
from ase.calculators.siesta.import_functions import \
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.siesta.parameters import PAOBasisBlock, Species
from ase.calculators.siesta.parameters import format_fdf
def _write_atomic_coordinates(self, fd, atoms):
    """Write atomic coordinates.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
    af = self.parameters.atomic_coord_format.lower()
    if af == 'xyz':
        self._write_atomic_coordinates_xyz(fd, atoms)
    elif af == 'zmatrix':
        self._write_atomic_coordinates_zmatrix(fd, atoms)
    else:
        raise RuntimeError('Unknown atomic_coord_format: {}'.format(af))