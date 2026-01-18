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
def _update_readiso_params(parameters, symbols):
    """ Deletes the ReadIso option from the route section as we
    write out the masses in the nuclear properties section
    instead of the ReadIso section.
    Ensures the masses array is the same length as the
    symbols array. This is necessary due to the way the
    ReadIso section is defined:
    The mass of each atom is listed on a separate line, in
    order of appearance in the molecule spec. A blank line
    indicates not to modify the mass for that atom.
    But you do not have to leave blank lines equal to the
    remaining atoms after you finsihed setting masses.
    E.g. if you had 10 masses and only want to set the mass
    for the first atom, you don't have to leave 9 blank lines
    after it.
    """
    parameters = _delete_readiso_param(parameters)
    if parameters.get('isolist') is not None:
        if len(parameters['isolist']) < len(symbols):
            for i in range(0, len(symbols) - len(parameters['isolist'])):
                parameters['isolist'].append(None)
        if all((m is None for m in parameters['isolist'])):
            parameters['isolist'] = None
    return parameters