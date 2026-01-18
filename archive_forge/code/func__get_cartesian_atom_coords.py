import re
import warnings
from collections.abc import Iterable
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.calculators.calculator import InputError, Calculator
from ase.calculators.gaussian import Gaussian
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import atomic_masses_iupac2016, chemical_symbols
from ase.io import ParseError
from ase.io.zmatrix import parse_zmatrix
from ase.units import Bohr, Hartree
def _get_cartesian_atom_coords(symbol, pos):
    """Returns the coordinates: pos as a list of floats if they
    are cartesian, and not in z-matrix format"""
    if len(pos) < 3 or (pos[0] == '0' and symbol != 'TV'):
        return
    elif len(pos) > 3:
        raise ParseError('ERROR: Gaussian input file could not be read as freeze codes are not supported. If using cartesian coordinates, these must be given as 3 numbers separated by whitespace.')
    else:
        try:
            return list(map(float, pos))
        except ValueError:
            raise ParseError('ERROR: Molecule specification inGaussian input file could not be read')