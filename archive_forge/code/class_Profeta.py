import numpy as np
import ase.units as u
from ase.vibrations.raman import Raman, RamanPhonons
from ase.vibrations.resonant_raman import ResonantRaman
from ase.calculators.excitation_list import polarizability
class Profeta(ResonantRaman):
    """Profeta type approximations.

    Reference
    ---------
    Mickael Profeta and Francesco Mauri
    Phys. Rev. B 63 (2000) 245415
    """

    def __init__(self, *args, **kwargs):
        self.set_approximation(kwargs.pop('approximation', 'Profeta'))
        self.nonresonant = kwargs.pop('nonresonant', True)
        ResonantRaman.__init__(self, *args, **kwargs)

    def set_approximation(self, value):
        approx = value.lower()
        if approx in ['profeta', 'placzek', 'p-p']:
            self._approx = value
        else:
            raise ValueError('Please use "Profeta", "Placzek" or "P-P".')

    def electronic_me_profeta_rcc(self, omega, gamma=0.1, energy_derivative=False):
        """Raman spectra in Profeta and Mauri approximation

        Returns
        -------
        Electronic matrix element, unit Angstrom^2
         """
        self.calculate_energies_and_modes()
        V_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        pre = 1.0 / (2 * self.delta)
        pre *= u.Hartree * u.Bohr

        def kappa_cc(me_pc, e_p, omega, gamma, form='v'):
            """Kappa tensor after Profeta and Mauri
            PRB 63 (2001) 245415"""
            k_cc = np.zeros((3, 3), dtype=complex)
            for p, me_c in enumerate(me_pc):
                me_cc = np.outer(me_c, me_c.conj())
                k_cc += me_cc / (e_p[p] - omega - 1j * gamma)
                if self.nonresonant:
                    k_cc += me_cc.conj() / (e_p[p] + omega + 1j * gamma)
            return k_cc
        mr = 0
        for a, i, r in zip(self.myindices, self.myxyz, self.myr):
            if not energy_derivative < 0:
                V_rcc[r] += pre * (kappa_cc(self.expm_rpc[mr], self.ex0E_p, omega, gamma, self.dipole_form) - kappa_cc(self.exmm_rpc[mr], self.ex0E_p, omega, gamma, self.dipole_form))
            if energy_derivative:
                V_rcc[r] += pre * (kappa_cc(self.ex0m_pc, self.expE_rp[mr], omega, gamma, self.dipole_form) - kappa_cc(self.ex0m_pc, self.exmE_rp[mr], omega, gamma, self.dipole_form))
            mr += 1
        self.comm.sum(V_rcc)
        return V_rcc

    def electronic_me_Qcc(self, omega, gamma):
        self.read()
        Vel_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        approximation = self.approximation.lower()
        if approximation == 'profeta':
            Vel_rcc += self.electronic_me_profeta_rcc(omega, gamma)
        elif approximation == 'placzek':
            Vel_rcc += self.electronic_me_profeta_rcc(omega, gamma, True)
        elif approximation == 'p-p':
            Vel_rcc += self.electronic_me_profeta_rcc(omega, gamma, -1)
        else:
            raise RuntimeError('Bug: call with {0} should not happen!'.format(self.approximation))
        return self.map_to_modes(Vel_rcc)