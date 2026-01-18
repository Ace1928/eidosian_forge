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
def _get_charge_mult(chgmult_section):
    """return a dict with the charge and multiplicity from
    a list chgmult_section that contains the charge and multiplicity
    line, read from a gaussian input file"""
    chgmult_match = _re_chgmult.match(str(chgmult_section))
    try:
        chgmult = chgmult_match.group(0).split()
        return {'charge': int(chgmult[0]), 'mult': int(chgmult[1])}
    except (IndexError, AttributeError):
        raise ParseError('ERROR: Could not read the charge and multiplicity from the Gaussian input file. These must be 2 integers separated with whitespace or a comma.')