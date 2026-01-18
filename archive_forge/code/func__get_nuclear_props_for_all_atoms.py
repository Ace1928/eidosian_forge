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
def _get_nuclear_props_for_all_atoms(nuclear_props):
    """ Returns the nuclear properties for all atoms as a dictionary,
    in the format needed for it to be added to the parameters dictionary."""
    params = {k + 'list': [] for k in _nuclear_prop_names}
    for dictionary in nuclear_props:
        for key, value in dictionary.items():
            params[key + 'list'].append(value)
    for key, array in params.items():
        values_set = False
        for value in array:
            if value is not None:
                values_set = True
        if not values_set:
            params[key] = None
    return params