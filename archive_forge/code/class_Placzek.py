import numpy as np
import ase.units as u
from ase.vibrations.raman import Raman, RamanPhonons
from ase.vibrations.resonant_raman import ResonantRaman
from ase.calculators.excitation_list import polarizability
class Placzek(ResonantRaman):
    """Raman spectra within the Placzek approximation."""

    def __init__(self, *args, **kwargs):
        self._approx = 'PlaczekAlpha'
        ResonantRaman.__init__(self, *args, **kwargs)

    def set_approximation(self, value):
        raise ValueError('Approximation can not be set.')

    def _signed_disps(self, sign):
        for a, i in zip(self.myindices, self.myxyz):
            yield self._disp(a, i, sign)

    def _read_exobjs(self, sign):
        return [disp.read_exobj() for disp in self._signed_disps(sign)]

    def read_excitations(self):
        """Read excitations from files written"""
        self.ex0E_p = None
        self.exm_r = self._read_exobjs(sign=-1)
        self.exp_r = self._read_exobjs(sign=1)

    def electronic_me_Qcc(self, omega, gamma=0):
        self.calculate_energies_and_modes()
        V_rcc = np.zeros((self.ndof, 3, 3), dtype=complex)
        pre = 1.0 / (2 * self.delta)
        pre *= u.Hartree * u.Bohr
        om = omega
        if gamma:
            om += 1j * gamma
        for i, r in enumerate(self.myr):
            V_rcc[r] = pre * (polarizability(self.exp_r[i], om, form=self.dipole_form, tensor=True) - polarizability(self.exm_r[i], om, form=self.dipole_form, tensor=True))
        self.comm.sum(V_rcc)
        return self.map_to_modes(V_rcc)