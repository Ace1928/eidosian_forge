import itertools
import numpy as np
from ase.lattice import bravais_lattices, UnconventionalLattice, bravais_names
from ase.cell import Cell
Yield all lattices defined by the length and angle grids.