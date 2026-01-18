import numpy as np
from ase.calculators.calculator import Calculator
from ase.data import atomic_numbers
from ase.utils import IOContext
from ase.geometry import get_distances
from ase.cell import Cell
class LJInteractionsGeneral:
    name = 'LJ-general'

    def __init__(self, sigmaqm, epsilonqm, sigmamm, epsilonmm, qm_molecule_size, mm_molecule_size=3, rc=np.Inf, width=1.0):
        """General Lennard-Jones type explicit interaction.

        sigmaqm: array
            Array of sigma-parameters which should have the length of the QM
            subsystem
        epsilonqm: array
            As sigmaqm, but for epsilon-paramaters
        sigmamm: Either array (A) or tuple (B)
            A (no counterions):
                Array of sigma-parameters with the length of the smallests
                repeating atoms-group (i.e. molecule) of the MM subsystem
            B (counterions):
                Tuple: (arr1, arr2), where arr1 is an array of sigmas with
                the length of counterions in the MM subsystem, and
                arr2 is the array from A.
        epsilonmm: array or tuple
            As sigmamm but for epsilon-parameters.
        qm_molecule_size: int
            number of atoms of the smallest repeating atoms-group (i.e.
            molecule) in the QM subsystem (often just the number of atoms
            in the QM subsystem)
        mm_molecule_size: int
            as qm_molecule_size but for the MM subsystem. Will be overwritten
            if counterions are present in the MM subsystem (via the CombineMM
            calculator)

        """
        self.sigmaqm = sigmaqm
        self.epsilonqm = epsilonqm
        self.sigmamm = sigmamm
        self.epsilonmm = epsilonmm
        self.qms = qm_molecule_size
        self.mms = mm_molecule_size
        self.rc = rc
        self.width = width
        self.combine_lj()

    def combine_lj(self):
        self.sigma, self.epsilon = combine_lj_lorenz_berthelot(self.sigmaqm, self.sigmamm, self.epsilonqm, self.epsilonmm)

    def calculate(self, qmatoms, mmatoms, shift):
        epsilon = self.epsilon
        sigma = self.sigma
        apm1 = self.mms
        mask1 = np.ones(len(mmatoms), dtype=bool)
        mask2 = mask1
        apm = (apm1,)
        sigma = (sigma,)
        epsilon = (epsilon,)
        if hasattr(mmatoms.calc, 'name'):
            if mmatoms.calc.name == 'combinemm':
                mask1 = mmatoms.calc.mask
                mask2 = ~mask1
                apm1 = mmatoms.calc.apm1
                apm2 = mmatoms.calc.apm2
                apm = (apm1, apm2)
                sigma = sigma[0]
                epsilon = epsilon[0]
        mask = (mask1, mask2)
        e_all = 0
        qmforces_all = np.zeros_like(qmatoms.positions)
        mmforces_all = np.zeros_like(mmatoms.positions)
        for n, m, eps, sig in zip(apm, mask, epsilon, sigma):
            mmpositions = self.update(qmatoms, mmatoms[m], n, shift)
            qmforces = np.zeros_like(qmatoms.positions)
            mmforces = np.zeros_like(mmatoms[m].positions)
            energy = 0.0
            qmpositions = qmatoms.positions.reshape((-1, self.qms, 3))
            for q, qmpos in enumerate(qmpositions):
                R00 = mmpositions[:, 0] - qmpos[0, :]
                d002 = (R00 ** 2).sum(1)
                d00 = d002 ** 0.5
                x1 = d00 > self.rc - self.width
                x2 = d00 < self.rc
                x12 = np.logical_and(x1, x2)
                y = (d00[x12] - self.rc + self.width) / self.width
                t = np.zeros(len(d00))
                t[x2] = 1.0
                t[x12] -= y ** 2 * (3.0 - 2.0 * y)
                dt = np.zeros(len(d00))
                dt[x12] -= 6.0 / self.width * y * (1.0 - y)
                for qa in range(len(qmpos)):
                    if ~np.any(eps[qa, :]):
                        continue
                    R = mmpositions - qmpos[qa, :]
                    d2 = (R ** 2).sum(2)
                    c6 = (sig[qa, :] ** 2 / d2) ** 3
                    c12 = c6 ** 2
                    e = 4 * eps[qa, :] * (c12 - c6)
                    energy += np.dot(e.sum(1), t)
                    f = t[:, None, None] * (24 * eps[qa, :] * (2 * c12 - c6) / d2)[:, :, None] * R
                    f00 = -(e.sum(1) * dt / d00)[:, None] * R00
                    mmforces += f.reshape((-1, 3))
                    qmforces[q * self.qms + qa, :] -= f.sum(0).sum(0)
                    qmforces[q * self.qms, :] -= f00.sum(0)
                    mmforces[::n, :] += f00
                e_all += energy
                qmforces_all += qmforces
                mmforces_all[m] += mmforces
        return (e_all, qmforces_all, mmforces_all)

    def update(self, qmatoms, mmatoms, n, shift):
        qmcenter = qmatoms.cell.diagonal() / 2
        positions = mmatoms.positions.reshape((-1, n, 3)) + shift
        distances = positions[:, 0] - qmcenter
        wrap(distances, mmatoms.cell.diagonal(), mmatoms.pbc)
        offsets = distances - positions[:, 0]
        positions += offsets[:, np.newaxis] + qmcenter
        return positions