import os
import re
from subprocess import call, TimeoutExpired
from copy import deepcopy
import numpy as np
from ase import Atoms
from ase.utils import workdir
from ase.units import Hartree, Bohr, Debye
from ase.calculators.singlepoint import SinglePointCalculator
def _write_geom(atoms, basis_spec):
    out = [' $DATA', atoms.get_chemical_formula(), 'C1']
    for i, atom in enumerate(atoms):
        out.append('{:<3} {:>3} {:20.13e} {:20.13e} {:20.13e}'.format(atom.symbol, atom.number, *atom.position))
        if basis_spec is not None:
            basis = basis_spec.get(i)
            if basis is None:
                basis = basis_spec.get(atom.symbol)
            if basis is None:
                raise ValueError('Could not find an appropriate basis set for atom number {}!'.format(i))
            out += [basis, '']
    out.append(' $END')
    return '\n'.join(out)