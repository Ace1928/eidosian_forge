from __future__ import annotations
import itertools
import os
import re
import warnings
from collections import UserDict
from typing import TYPE_CHECKING, Any
import numpy as np
import spglib
from monty.io import zopen
from monty.json import MSONable
from monty.serialization import loadfn
from pymatgen.core.structure import Structure
from pymatgen.io.vasp import Vasprun
from pymatgen.io.vasp.inputs import Incar, Kpoints, Potcar
from pymatgen.symmetry.bandstructure import HighSymmKpath
from pymatgen.util.due import Doi, due
def _get_nbands(self, structure: Structure):
    """Get number of bands."""
    if self.get('basisfunctions') is None:
        raise OSError('No basis functions are provided. The program cannot calculate nbands.')
    basis_functions: list[str] = []
    for string_basis in self['basisfunctions']:
        string_basis_raw = string_basis.strip().split(' ')
        while '' in string_basis_raw:
            string_basis_raw.remove('')
        for _idx in range(int(structure.composition.element_composition[string_basis_raw[0]])):
            basis_functions.extend(string_basis_raw[1:])
    no_basis_functions = 0
    for basis in basis_functions:
        if 's' in basis:
            no_basis_functions = no_basis_functions + 1
        elif 'p' in basis:
            no_basis_functions = no_basis_functions + 3
        elif 'd' in basis:
            no_basis_functions = no_basis_functions + 5
        elif 'f' in basis:
            no_basis_functions = no_basis_functions + 7
    return int(no_basis_functions)