import os
import os.path as op
import subprocess
import shutil
import numpy as np
from ase.units import Bohr, Hartree
import ase.data
from ase.calculators.calculator import FileIOCalculator, ReadError
from ase.calculators.calculator import Parameters, all_changes
from ase.calculators.calculator import equal
import ase.io
from .demon_io import parse_xray
def _write_basis(self, fd, atoms, basis={}, string='BASIS'):
    """Write basis set, ECPs, AUXIS, or AUGMENT basis

        Parameters:
        - f:     An open file object.
        - atoms: An atoms object.
        - basis: A dictionary specifying the basis set
        - string: 'BASIS', 'ECP','AUXIS' or 'AUGMENT'
        """
    line = '{0}'.format(string).ljust(10)
    if 'all' in basis:
        default_basis = basis['all']
        line += '({0})'.format(default_basis).rjust(16)
    fd.write(line)
    fd.write('\n')
    chemical_symbols = atoms.get_chemical_symbols()
    chemical_symbols_set = set(chemical_symbols)
    for i in range(chemical_symbols_set.__len__()):
        symbol = chemical_symbols_set.pop()
        if symbol in basis:
            line = '{0}'.format(symbol).ljust(10)
            line += '({0})'.format(basis[symbol]).rjust(16)
            fd.write(line)
            fd.write('\n')
    for i in range(len(atoms)):
        if i in basis:
            symbol = str(chemical_symbols[i])
            symbol += str(i + 1)
            line = '{0}'.format(symbol).ljust(10)
            line += '({0})'.format(basis[i]).rjust(16)
            fd.write(line)
            fd.write('\n')