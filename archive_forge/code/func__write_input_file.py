import os
import re
import numpy as np
from ase import Atoms
from ase.io import read
from ase.io.dmol import write_dmol_car, write_dmol_incoor
from ase.units import Hartree, Bohr
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
def _write_input_file(self, fd):
    fd.write('%-32s %s\n' % ('calculate', 'gradient'))
    fd.write('%-32s %s\n' % ('print', 'eigval_last_it'))
    for key, value in self.parameters.items():
        if isinstance(value, str):
            fd.write('%-32s %s\n' % (key, value))
        elif isinstance(value, (list, tuple)):
            for val in value:
                fd.write('%-32s %s\n' % (key, val))
        else:
            fd.write('%-32s %r\n' % (key, value))