import os
import re
from subprocess import call, TimeoutExpired
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.utils import workdir
from ase.units import Hartree, Bohr, Debye
from ase.calculators.singlepoint import SinglePointCalculator
def _write_ecp(atoms, ecp):
    out = [' $ECP']
    for i, symbol in enumerate(atoms.symbols):
        if i in ecp:
            out.append(ecp[i])
        elif symbol in ecp:
            out.append(ecp[symbol])
        else:
            raise ValueError('Could not find an appropriate ECP for atom number {}!'.format(i))
    out.append(' $END')
    return '\n'.join(out)