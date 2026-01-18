import os
import subprocess
from warnings import warn
import numpy as np
from ase.calculators.calculator import (Calculator, FileIOCalculator,
from ase.io import write
from ase.io.vasp import write_vasp
from ase.parallel import world
from ase.units import Bohr, Hartree
def _generate_command(self):
    command = self.command.split()
    if any(self.atoms.pbc):
        command.append(self.label + '.POSCAR')
    else:
        command.append(self.label + '.xyz')
    if not self.custom_damp:
        xc = self.parameters.get('xc')
        if xc is None:
            xc = 'pbe'
        command += ['-func', xc.lower()]
    for arg in self.dftd3_flags:
        if self.parameters.get(arg):
            command.append('-' + arg)
    if any(self.atoms.pbc):
        command.append('-pbc')
    command += ['-cnthr', str(self.parameters['cnthr'] / Bohr)]
    command += ['-cutoff', str(self.parameters['cutoff'] / Bohr)]
    if not self.parameters['old']:
        command.append('-' + self.parameters['damping'])
    return command