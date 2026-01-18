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
def _get_route_params(line):
    """Reads keywords and values from a line in
    a Gaussian input file's route section,
    and returns them as a dictionary"""
    method_basis_match = _re_method_basis.match(line)
    if method_basis_match:
        params = {}
        ase_gen_comment = '! ASE formatted method and basis'
        if method_basis_match.group(5) == ase_gen_comment:
            params['method'] = method_basis_match.group(1).strip().lower()
            params['basis'] = method_basis_match.group(2).strip().lower()
            if method_basis_match.group(4):
                params['fitting_basis'] = method_basis_match.group(4).strip().lower()
            return params
    return _get_key_value_pairs(line)