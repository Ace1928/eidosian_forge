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
def _format_addsec(addsec):
    """Format addsec string as a list of lines to be added to the gaussian
     input file."""
    out = []
    if addsec is not None:
        out.append('')
        if isinstance(addsec, str):
            out.append(addsec)
        elif isinstance(addsec, Iterable):
            out += list(addsec)
    return out