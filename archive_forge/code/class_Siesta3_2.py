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
class Siesta3_2(Siesta):

    def __init__(self, *args, **kwargs):
        warnings.warn("The Siesta3_2 calculator class will no longer be supported. Use 'ase.calculators.siesta.Siesta in stead. If using the ASE interface with SIESTA 3.2 you must explicitly include the keywords 'SpinPolarized', 'NonCollinearSpin' and 'SpinOrbit' if needed.", np.VisibleDeprecationWarning)
        Siesta.__init__(self, *args, **kwargs)