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
def _get_gaussian_in_sections(fd):
    """ Reads a gaussian input file and returns
    a dictionary of the sections of the file - each dictionary
    value is a list of strings - each string is a line in that
    section. """
    route_section = False
    atoms_section = False
    atoms_saved = False
    gaussian_sections = {'link0': [], 'route': [], 'charge_mult': [], 'mol_spec': [], 'extra': []}
    for line in fd:
        line = line.strip(' ')
        link0_match = _re_link0.match(line)
        output_type_match = _re_output_type.match(line)
        chgmult_match = _re_chgmult.match(line)
        if link0_match:
            gaussian_sections['link0'].append(line)
        elif line == '\n' and (route_section or atoms_section):
            route_section = False
            atoms_section = False
        elif output_type_match or route_section:
            route_section = True
            gaussian_sections['route'].append(line)
        elif chgmult_match:
            gaussian_sections['charge_mult'] = line
            atoms_section = True
        elif atoms_section:
            gaussian_sections['mol_spec'].append(line)
            atoms_saved = True
        elif atoms_saved:
            gaussian_sections['extra'].append(line)
    return gaussian_sections