from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('body-centred tetragonal', 'tetragonal', 'tetragonal', 'tI', 'ac', [['BCT1', 'GMNPXZZ1', 'GXMGZPNZ1M,XP', None], ['BCT2', 'GNPSS1XYY1Z', 'GXYSGZS1NPY1Z,XP', None]])
class BCT(BravaisLattice):
    conventional_cls = 'TET'
    conventional_cellmap = _fcc_map

    def __init__(self, a, c):
        BravaisLattice.__init__(self, a=a, c=c)

    def _cell(self, a, c):
        return 0.5 * np.array([[-a, a, c], [a, -a, c], [a, a, -c]])

    def _variant_name(self, a, c):
        return 'BCT1' if c < a else 'BCT2'

    def _special_points(self, a, c, variant):
        a2 = a * a
        c2 = c * c
        assert variant.name in self.variants
        if variant.name == 'BCT1':
            eta = 0.25 * (1 + c2 / a2)
            points = [[0, 0, 0], [-0.5, 0.5, 0.5], [0.0, 0.5, 0.0], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5], [eta, eta, -eta], [-eta, 1 - eta, eta]]
        else:
            eta = 0.25 * (1 + a2 / c2)
            zeta = 0.5 * a2 / c2
            points = [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.25, 0.25, 0.25], [-eta, eta, eta], [eta, 1 - eta, -eta], [0.0, 0.0, 0.5], [-zeta, zeta, 0.5], [0.5, 0.5, -zeta], [0.5, 0.5, -0.5]]
        return points