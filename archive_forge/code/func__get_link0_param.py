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
def _get_link0_param(link0_match):
    """Gets link0 keyword and option from a re.Match and returns them
    in a dictionary format"""
    value = link0_match.group(2)
    if value is not None:
        value = value.strip()
    else:
        value = ''
    return {link0_match.group(1).lower().strip(): value.lower()}