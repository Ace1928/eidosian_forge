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
def _get_readiso_info(line, parameters):
    """Reads the temperature, pressure and scale from the first line
    of a ReadIso section of a Gaussian input file. Returns these in
    a dictionary."""
    readiso_params = {}
    if _get_readiso_param(parameters)[0] is not None:
        line = line.replace('[', '').replace(']', '')
        tokens = line.strip().split()
        try:
            readiso_params['temperature'] = tokens[0]
            readiso_params['pressure'] = tokens[1]
            readiso_params['scale'] = tokens[2]
        except IndexError:
            pass
    if readiso_params != {}:
        return readiso_params