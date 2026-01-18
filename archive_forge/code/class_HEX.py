from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('primitive hexagonal', 'hexagonal', 'hexagonal', 'hP', 'ac', [['HEX', 'GMKALH', 'GMKGALHA,LM,KH', sc_special_points['hexagonal']]])
class HEX(BravaisLattice):
    conventional_cls = 'HEX'
    conventional_cellmap = _identity

    def __init__(self, a, c):
        BravaisLattice.__init__(self, a=a, c=c)

    def _cell(self, a, c):
        x = 0.5 * np.sqrt(3)
        return np.array([[0.5 * a, -x * a, 0], [0.5 * a, x * a, 0], [0.0, 0.0, c]])