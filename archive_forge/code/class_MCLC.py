from abc import abstractmethod, ABC
import functools
import warnings
import numpy as np
from typing import Dict, List
from ase.cell import Cell
from ase.build.bulk import bulk as newbulk
from ase.dft.kpoints import parse_path_string, sc_special_points, BandPath
from ase.utils import pbc2pbc
@bravaisclass('base-centred monoclinic', 'monoclinic', 'monoclinic', 'mC', ('a', 'b', 'c', 'alpha'), [['MCLC1', 'GNN1FF1F2F3II1LMXX1X2YY1Z', 'GYFLI,I1ZF1,YX1,XGN,MG', None], ['MCLC2', 'GNN1FF1F2F3II1LMXX1X2YY1Z', 'GYFLI,I1ZF1,NGM', None], ['MCLC3', 'GFF1F2HH1H2IMNN1XYY1Y2Y3Z', 'GYFHZIF1,H1Y1XGN,MG', None], ['MCLC4', 'GFF1F2HH1H2IMNN1XYY1Y2Y3Z', 'GYFHZI,H1Y1XGN,MG', None], ['MCLC5', 'GFF1F2HH1H2II1LMNN1XYY1Y2Y3Z', 'GYFLI,I1ZHF1,H1Y1XGN,MG', None]])
class MCLC(BravaisLattice):
    conventional_cls = 'MCL'
    conventional_cellmap = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 1]])

    def __init__(self, a, b, c, alpha):
        check_mcl(a, b, c, alpha)
        BravaisLattice.__init__(self, a=a, b=b, c=c, alpha=alpha)

    def _cell(self, a, b, c, alpha):
        alpha *= np.pi / 180
        return np.array([[0.5 * a, 0.5 * b, 0], [-0.5 * a, 0.5 * b, 0], [0, c * np.cos(alpha), c * np.sin(alpha)]])

    def _variant_name(self, a, b, c, alpha):
        check_mcl(a, b, c, alpha)
        a2 = a * a
        b2 = b * b
        cosa = np.cos(alpha * _degrees)
        sina = np.sin(alpha * _degrees)
        sina2 = sina ** 2
        cell = self.tocell()
        lengths_angles = Cell(cell.reciprocal()).cellpar()
        kgamma = lengths_angles[-1]
        eps = self._eps
        if abs(kgamma - 90) < eps:
            variant = 2
        elif kgamma > 90:
            variant = 1
        elif kgamma < 90:
            num = b * cosa / c + b2 * sina2 / a2
            if abs(num - 1) < eps:
                variant = 4
            elif num < 1:
                variant = 3
            else:
                variant = 5
        variant = 'MCLC' + str(variant)
        return variant

    def _special_points(self, a, b, c, alpha, variant):
        variant = int(variant.name[-1])
        a2 = a * a
        b2 = b * b
        cosa = np.cos(alpha * _degrees)
        sina = np.sin(alpha * _degrees)
        sina2 = sina ** 2
        if variant == 1 or variant == 2:
            zeta = (2 - b * cosa / c) / (4 * sina2)
            eta = 0.5 + 2 * zeta * c * cosa / b
            psi = 0.75 - a2 / (4 * b2 * sina * sina)
            phi = psi + (0.75 - psi) * b * cosa / c
            points = [[0, 0, 0], [0.5, 0, 0], [0, -0.5, 0], [1 - zeta, 1 - zeta, 1 - eta], [zeta, zeta, eta], [-zeta, -zeta, 1 - eta], [1 - zeta, -zeta, 1 - eta], [phi, 1 - phi, 0.5], [1 - phi, phi - 1, 0.5], [0.5, 0.5, 0.5], [0.5, 0, 0.5], [1 - psi, psi - 1, 0], [psi, 1 - psi, 0], [psi - 1, -psi, 0], [0.5, 0.5, 0], [-0.5, -0.5, 0], [0, 0, 0.5]]
        elif variant == 3 or variant == 4:
            mu = 0.25 * (1 + b2 / a2)
            delta = b * c * cosa / (2 * a2)
            zeta = mu - 0.25 + (1 - b * cosa / c) / (4 * sina2)
            eta = 0.5 + 2 * zeta * c * cosa / b
            phi = 1 + zeta - 2 * mu
            psi = eta - 2 * delta
            points = [[0, 0, 0], [1 - phi, 1 - phi, 1 - psi], [phi, phi - 1, psi], [1 - phi, -phi, 1 - psi], [zeta, zeta, eta], [1 - zeta, -zeta, 1 - eta], [-zeta, -zeta, 1 - eta], [0.5, -0.5, 0.5], [0.5, 0, 0.5], [0.5, 0, 0], [0, -0.5, 0], [0.5, -0.5, 0], [mu, mu, delta], [1 - mu, -mu, -delta], [-mu, -mu, -delta], [mu, mu - 1, delta], [0, 0, 0.5]]
        elif variant == 5:
            zeta = 0.25 * (b2 / a2 + (1 - b * cosa / c) / sina2)
            eta = 0.5 + 2 * zeta * c * cosa / b
            mu = 0.5 * eta + b2 / (4 * a2) - b * c * cosa / (2 * a2)
            nu = 2 * mu - zeta
            omega = (4 * nu - 1 - b2 * sina2 / a2) * c / (2 * b * cosa)
            delta = zeta * c * cosa / b + omega / 2 - 0.25
            rho = 1 - zeta * a2 / b2
            points = [[0, 0, 0], [nu, nu, omega], [1 - nu, 1 - nu, 1 - omega], [nu, nu - 1, omega], [zeta, zeta, eta], [1 - zeta, -zeta, 1 - eta], [-zeta, -zeta, 1 - eta], [rho, 1 - rho, 0.5], [1 - rho, rho - 1, 0.5], [0.5, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0, 0], [0, -0.5, 0], [0.5, -0.5, 0], [mu, mu, delta], [1 - mu, -mu, -delta], [-mu, -mu, -delta], [mu, mu - 1, delta], [0, 0, 0.5]]
        return points