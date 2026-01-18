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
def _get_all_route_params(route_section):
    """ Given a string: route_section which contains the route
    section of a gaussian input file, returns a dictionary of
    keywords and values"""
    parameters = {}
    for line in route_section:
        output_type_match = _re_output_type.match(line)
        if not parameters.get('output_type') and output_type_match:
            line = line.strip(output_type_match.group(0))
            parameters.update({'output_type': output_type_match.group(1).lower()})
        route_params = _get_route_params(line)
        if route_params is not None:
            parameters.update(route_params)
    return parameters