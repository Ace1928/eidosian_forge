from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('primitive oblique', 'monoclinic', None, 'mp', ('a', 'b', 'alpha'), [['OBL', 'GYHCH1X', 'GYHCH1XG', None]], ndim=2)
class OBL(BravaisLattice):

    def __init__(self, a, b, alpha, **kwargs):
        BravaisLattice.__init__(self, a=a, b=b, alpha=alpha, **kwargs)

    def _cell(self, a, b, alpha):
        cosa = np.cos(alpha * _degrees)
        sina = np.sin(alpha * _degrees)
        return np.array([[a, 0, 0], [b * cosa, b * sina, 0], [0.0, 0.0, 0.0]])

    def _special_points(self, a, b, alpha, variant):
        if alpha > 90:
            _alpha = 180 - alpha
            a, b = (b, a)
        else:
            _alpha = alpha
        cosa = np.cos(_alpha * _degrees)
        eta = (1 - a * cosa / b) / (2 * np.sin(_alpha * _degrees) ** 2)
        nu = 0.5 - eta * b * cosa / a
        points = [[0, 0, 0], [0, 0.5, 0], [eta, 1 - nu, 0], [0.5, 0.5, 0], [1 - eta, nu, 0], [0.5, 0, 0]]
        if alpha > 90:
            op = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
            points = np.dot(points, op.T)
        return points