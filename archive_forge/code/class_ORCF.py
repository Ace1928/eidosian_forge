from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('face-centred orthorhombic', 'orthorhombic', 'orthorhombic', 'oF', 'abc', [['ORCF1', 'GAA1LTXX1YZ', 'GYTZGXA1Y,TX1,XAZ,LG', None], ['ORCF2', 'GCC1DD1LHH1XYZ', 'GYCDXGZD1HC,C1Z,XH1,HY,LG', None], ['ORCF3', 'GAA1LTXX1YZ', 'GYTZGXA1Y,XAZ,LG', None]])
class ORCF(Orthorhombic):
    conventional_cls = 'ORC'
    conventional_cellmap = _bcc_map

    def _cell(self, a, b, c):
        return 0.5 * np.array([[0, b, c], [a, 0, c], [a, b, 0]])

    def _special_points(self, a, b, c, variant):
        a2 = a * a
        b2 = b * b
        c2 = c * c
        xminus = 0.25 * (1 + a2 / b2 - a2 / c2)
        xplus = 0.25 * (1 + a2 / b2 + a2 / c2)
        if variant.name == 'ORCF1' or variant.name == 'ORCF3':
            zeta = xminus
            eta = xplus
            points = [[0, 0, 0], [0.5, 0.5 + zeta, zeta], [0.5, 0.5 - zeta, 1 - zeta], [0.5, 0.5, 0.5], [1.0, 0.5, 0.5], [0.0, eta, eta], [1.0, 1 - eta, 1 - eta], [0.5, 0, 0.5], [0.5, 0.5, 0]]
        else:
            assert variant.name == 'ORCF2'
            phi = 0.25 * (1 + c2 / b2 - c2 / a2)
            delta = 0.25 * (1 + b2 / a2 - b2 / c2)
            eta = xminus
            points = [[0, 0, 0], [0.5, 0.5 - eta, 1 - eta], [0.5, 0.5 + eta, eta], [0.5 - delta, 0.5, 1 - delta], [0.5 + delta, 0.5, delta], [0.5, 0.5, 0.5], [1 - phi, 0.5 - phi, 0.5], [phi, 0.5 + phi, 0.5], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]
        return points

    def _variant_name(self, a, b, c):
        diff = 1.0 / (a * a) - 1.0 / (b * b) - 1.0 / (c * c)
        if abs(diff) < self._eps:
            return 'ORCF3'
        return 'ORCF1' if diff > 0 else 'ORCF2'