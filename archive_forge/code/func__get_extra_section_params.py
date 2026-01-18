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
def _get_extra_section_params(extra_section, parameters, atoms):
    """ Takes a list of strings: extra_section, which contains
    the 'extra' lines in a gaussian input file. Also takes the parameters
    that have been read so far, and the atoms that have been read from the
    file.
    Returns an updated version of the parameters dict, containing details from
    this extra section. This may include the basis set definition or filename,
    and/or the readiso section."""
    new_parameters = deepcopy(parameters)
    basis_set = ''
    readiso = _get_readiso_param(new_parameters)[0]
    count_iso = 0
    readiso_masses = []
    for line in extra_section:
        if line.split():
            if line.split()[0] == '!':
                continue
            line = line.strip().split('!')[0]
        if len(line) > 0 and line[0] == '@':
            new_parameters['basisfile'] = line
        elif readiso and count_iso < len(atoms.symbols) + 1:
            if count_iso == 0 and line != '\n':
                readiso_info = _get_readiso_info(line, new_parameters)
                if readiso_info is not None:
                    new_parameters.update(readiso_info)
                readiso_masses = []
                count_iso += 1
            elif count_iso > 0:
                try:
                    readiso_masses.append(float(line))
                except ValueError:
                    readiso_masses.append(None)
                count_iso += 1
        elif new_parameters.get('basis', '') == 'gen' or 'gen' in new_parameters:
            if line.strip() != '':
                basis_set += line + '\n'
    if readiso:
        new_parameters['isolist'] = readiso_masses
        new_parameters = _update_readiso_params(new_parameters, atoms.symbols)
    if basis_set != '':
        new_parameters['basis_set'] = basis_set
        new_parameters['basis'] = 'gen'
        new_parameters.pop('gen', None)
    return new_parameters