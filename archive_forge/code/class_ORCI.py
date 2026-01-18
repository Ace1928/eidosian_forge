from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('body-centred orthorhombic', 'orthorhombic', 'orthorhombic', 'oI', 'abc', [['ORCI', 'GLL1L2RSTWXX1YY1Z', 'GXLTWRX1ZGYSW,L1Y,Y1Z', None]])
class ORCI(Orthorhombic):
    conventional_cls = 'ORC'
    conventional_cellmap = _fcc_map

    def _cell(self, a, b, c):
        return 0.5 * np.array([[-a, b, c], [a, -b, c], [a, b, -c]])

    def _special_points(self, a, b, c, variant):
        a2 = a ** 2
        b2 = b ** 2
        c2 = c ** 2
        zeta = 0.25 * (1 + a2 / c2)
        eta = 0.25 * (1 + b2 / c2)
        delta = 0.25 * (b2 - a2) / c2
        mu = 0.25 * (a2 + b2) / c2
        points = [[0.0, 0.0, 0.0], [-mu, mu, 0.5 - delta], [mu, -mu, 0.5 + delta], [0.5 - delta, 0.5 + delta, -mu], [0, 0.5, 0], [0.5, 0, 0], [0.0, 0.0, 0.5], [0.25, 0.25, 0.25], [-zeta, zeta, zeta], [zeta, 1 - zeta, -zeta], [eta, -eta, eta], [1 - eta, eta, -eta], [0.5, 0.5, -0.5]]
        return points