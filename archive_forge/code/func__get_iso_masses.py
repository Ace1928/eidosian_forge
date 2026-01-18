import copy
from io import StringIO
import numpy as np
import pytest
from ase.atoms import Atoms
from ase.calculators.calculator import InputError
from ase.calculators.gaussian import Gaussian
from ase.io import ParseError
from ase.io.gaussian import (_get_atoms_info, _get_cartesian_atom_coords,
def _get_iso_masses(atoms):
    if atoms.calc.parameters.get('isolist'):
        return list(atoms.calc.parameters['isolist'])