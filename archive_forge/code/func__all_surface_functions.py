from math import sqrt
from operator import itemgetter
import numpy as np
from ase.atom import Atom
from ase.atoms import Atoms
from ase.data import reference_states, atomic_numbers
from ase.lattice.cubic import FaceCenteredCubic
def _all_surface_functions():
    d = {}
    for func in [fcc100, fcc110, bcc100, bcc110, bcc111, fcc111, hcp0001, hcp10m10, diamond100, diamond111, fcc111, mx2, graphene]:
        d[func.__name__] = func
    return d