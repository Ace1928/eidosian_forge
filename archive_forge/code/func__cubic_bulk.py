from math import sqrt
from ase.atoms import Atoms
from ase.symbols import string2symbols
from ase.data import reference_states, atomic_numbers, chemical_symbols
from ase.utils import plural
def _cubic_bulk(name, crystalstructure, a):
    if crystalstructure == 'fcc':
        atoms = Atoms(4 * name, cell=(a, a, a), pbc=True, scaled_positions=[(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)])
    elif crystalstructure == 'diamond':
        atoms = _cubic_bulk(2 * name, 'zincblende', a)
    elif crystalstructure == 'zincblende':
        atoms = Atoms(4 * name, cell=(a, a, a), pbc=True, scaled_positions=[(0, 0, 0), (0.25, 0.25, 0.25), (0, 0.5, 0.5), (0.25, 0.75, 0.75), (0.5, 0, 0.5), (0.75, 0.25, 0.75), (0.5, 0.5, 0), (0.75, 0.75, 0.25)])
    elif crystalstructure == 'rocksalt':
        atoms = Atoms(4 * name, cell=(a, a, a), pbc=True, scaled_positions=[(0, 0, 0), (0.5, 0, 0), (0, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0, 0.5), (0, 0, 0.5), (0.5, 0.5, 0), (0, 0.5, 0)])
    else:
        raise incompatible_cell(want='cubic', have=crystalstructure)
    return atoms