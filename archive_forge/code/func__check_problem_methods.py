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
def _check_problem_methods(method):
    """ Check method string for problem methods and warn appropriately"""
    if method.lower() in _problem_methods:
        warnings.warn('The requested method, {}, is a composite method. Composite methods do not have well-defined potential energy surfaces, so the energies, forces, and other properties returned by ASE may not be meaningful, or they may correspond to a different geometry than the one provided. Please use these methods with caution.'.format(method))