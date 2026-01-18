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
def _convert_to_symbol(string):
    """Converts an input string into a format
    that can be input to the 'symbol' parameter of an
    ASE Atom object (can be a chemical symbol (str)
    or an atomic number (int)).
    This is achieved by either stripping any
    integers from the string, or converting a string
    containing an atomic number to integer type"""
    symbol = _validate_symbol_string(string)
    if symbol.isnumeric():
        atomic_number = int(symbol)
        symbol = chemical_symbols[atomic_number]
    else:
        match = re.match('([A-Za-z]+)', symbol)
        symbol = match.group(1)
    return symbol