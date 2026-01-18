import pytest
import numpy as np
from ase.transport.tools import dagger, normalize
from ase.dft.kpoints import monkhorst_pack
from ase.build import molecule
from ase.io.cube import read_cube
from ase.lattice import CUB, FCC, BCC, TET, BCT, ORC, ORCF, ORCI, ORCC, HEX, \
from ase.dft.wannier import gram_schmidt, lowdin, random_orthogonal_matrix, \
class Paraboloid:

    def __init__(self, pos=(10.0, 10.0, 10.0), shift=1.0):
        self.pos = np.array(pos, dtype=complex)
        self.shift = shift

    def get_gradients(self):
        return 2 * self.pos

    def step(self, dF, updaterot=True, updatecoeff=True):
        self.pos -= dF

    def get_functional_value(self):
        return np.sum(self.pos ** 2) + self.shift