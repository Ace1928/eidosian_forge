from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('primitive rhombohedral', 'hexagonal', 'rhombohedral', 'hR', ('a', 'alpha'), [['RHL1', 'GBB1FLL1PP1P2QXZ', 'GLB1,BZGX,QFP1Z,LP', None], ['RHL2', 'GFLPP1QQ1Z', 'GPZQGFP1Q1LZ', None]])
class RHL(BravaisLattice):
    conventional_cls = 'RHL'
    conventional_cellmap = _identity

    def __init__(self, a, alpha):
        if alpha >= 120:
            raise UnconventionalLattice('Need alpha < 120 degrees, got {}'.format(alpha))
        BravaisLattice.__init__(self, a=a, alpha=alpha)

    def _cell(self, a, alpha):
        alpha *= np.pi / 180
        acosa = a * np.cos(alpha)
        acosa2 = a * np.cos(0.5 * alpha)
        asina2 = a * np.sin(0.5 * alpha)
        acosfrac = acosa / acosa2
        xx = 1 - acosfrac ** 2
        assert xx > 0.0
        return np.array([[acosa2, -asina2, 0], [acosa2, asina2, 0], [a * acosfrac, 0, a * xx ** 0.5]])

    def _variant_name(self, a, alpha):
        return 'RHL1' if alpha < 90 else 'RHL2'

    def _special_points(self, a, alpha, variant):
        if variant.name == 'RHL1':
            cosa = np.cos(alpha * _degrees)
            eta = (1 + 4 * cosa) / (2 + 4 * cosa)
            nu = 0.75 - 0.5 * eta
            points = [[0, 0, 0], [eta, 0.5, 1 - eta], [0.5, 1 - eta, eta - 1], [0.5, 0.5, 0], [0.5, 0, 0], [0, 0, -0.5], [eta, nu, nu], [1 - nu, 1 - nu, 1 - eta], [nu, nu, eta - 1], [1 - nu, nu, 0], [nu, 0, -nu], [0.5, 0.5, 0.5]]
        else:
            eta = 1 / (2 * np.tan(alpha * _degrees / 2) ** 2)
            nu = 0.75 - 0.5 * eta
            points = [[0, 0, 0], [0.5, -0.5, 0], [0.5, 0, 0], [1 - nu, -nu, 1 - nu], [nu, nu - 1, nu - 1], [eta, eta, eta], [1 - eta, -eta, -eta], [0.5, -0.5, 0.5]]
        return points