import sys
import numpy as np
import ase.units as u
from ase.parallel import world, paropen, parprint
from ase.vibrations import Vibrations
from ase.vibrations.raman import Raman, RamanCalculatorBase
class LrResonantRaman(ResonantRaman):
    """Resonant Raman for linear response

    Quick and dirty approach to enable loading of LrTDDFT calculations
    """

    def read_excitations(self):
        eq_disp = self._eq_disp()
        ex0_object = eq_disp.read_exobj()
        eu = ex0_object.energy_to_eV_scale
        matching = frozenset(ex0_object.kss)

        def append(lst, disp, matching):
            exo = disp.read_exobj()
            lst.append(exo)
            matching = matching.intersection(exo.kss)
            return matching
        exm_object_list = []
        exp_object_list = []
        for a in self.indices:
            for i in 'xyz':
                disp1 = self._disp(a, i, -1)
                disp2 = self._disp(a, i, 1)
                matching = append(exm_object_list, disp1, matching)
                matching = append(exp_object_list, disp2, matching)

        def select(exl, matching):
            exl.diagonalize(**self.exkwargs)
            mlist = list(exl)
            return mlist
        ex0 = select(ex0_object, matching)
        exm = []
        exp = []
        r = 0
        for a in self.indices:
            for i in 'xyz':
                exm.append(select(exm_object_list[r], matching))
                exp.append(select(exp_object_list[r], matching))
                r += 1
        self.ex0E_p = np.array([ex.energy * eu for ex in ex0])
        self.ex0m_pc = np.array([ex.get_dipole_me(form=self.dipole_form) for ex in ex0]) * u.Bohr
        self.exF_rp = []
        exmE_rp = []
        expE_rp = []
        exmm_rpc = []
        expm_rpc = []
        r = 0
        for a in self.indices:
            for i in 'xyz':
                exmE_rp.append([em.energy for em in exm[r]])
                expE_rp.append([ep.energy for ep in exp[r]])
                self.exF_rp.append([em.energy - ep.energy for ep, em in zip(exp[r], exm[r])])
                exmm_rpc.append([ex.get_dipole_me(form=self.dipole_form) for ex in exm[r]])
                expm_rpc.append([ex.get_dipole_me(form=self.dipole_form) for ex in exp[r]])
                r += 1
        self.exmE_rp = np.array(exmE_rp) * eu
        self.expE_rp = np.array(expE_rp) * eu
        self.exF_rp = np.array(self.exF_rp) * eu / 2 / self.delta
        self.exmm_rpc = np.array(exmm_rpc) * u.Bohr
        self.expm_rpc = np.array(expm_rpc) * u.Bohr