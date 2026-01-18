from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('primitive hexagonal', 'hexagonal', None, 'hp', 'a', [['HEX2D', 'GMK', 'GMKG', get_subset_points('GMK', sc_special_points['hexagonal'])]], ndim=2)
class HEX2D(BravaisLattice):

    def __init__(self, a, **kwargs):
        BravaisLattice.__init__(self, a=a, **kwargs)

    def _cell(self, a):
        x = 0.5 * np.sqrt(3)
        return np.array([[a, 0, 0], [-0.5 * a, x * a, 0], [0.0, 0.0, 0.0]])