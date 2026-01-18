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
def _format_method_basis(output_type, method, basis, fitting_basis):
    output_string = ''
    if basis and method and fitting_basis:
        output_string = '{} {}/{}/{} ! ASE formatted method and basis'.format(output_type, method, basis, fitting_basis)
    elif basis and method:
        output_string = '{} {}/{} ! ASE formatted method and basis'.format(output_type, method, basis)
    else:
        output_string = '{}'.format(output_type)
        for value in [method, basis]:
            if value is not None:
                output_string += ' {}'.format(value)
    return output_string