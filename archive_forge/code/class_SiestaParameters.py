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
class SiestaParameters(Parameters):
    """Parameters class for the calculator.
    Documented in BaseSiesta.__init__

    """

    def __init__(self, label='siesta', mesh_cutoff=200 * Ry, energy_shift=100 * meV, kpts=None, xc='LDA', basis_set='DZP', spin='non-polarized', species=tuple(), pseudo_qualifier=None, pseudo_path=None, symlink_pseudos=None, atoms=None, restart=None, fdf_arguments=None, atomic_coord_format='xyz', bandpath=None):
        kwargs = locals()
        kwargs.pop('self')
        Parameters.__init__(self, **kwargs)