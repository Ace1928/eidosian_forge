import numpy as np
from ase.calculators.calculator import Calculator, all_properties
from ase.calculators.calculator import PropertyNotImplementedError
class SinglePointDFTCalculator(SinglePointCalculator):

    def __init__(self, atoms, efermi=None, bzkpts=None, ibzkpts=None, bz2ibz=None, **results):
        self.bz_kpts = bzkpts
        self.ibz_kpts = ibzkpts
        self.bz2ibz = bz2ibz
        self.eFermi = efermi
        SinglePointCalculator.__init__(self, atoms, **results)
        self.kpts = None

    def get_fermi_level(self):
        """Return the Fermi-level(s)."""
        return self.eFermi

    def get_bz_to_ibz_map(self):
        return self.bz2ibz

    def get_bz_k_points(self):
        """Return the k-points."""
        return self.bz_kpts

    def get_number_of_spins(self):
        """Return the number of spins in the calculation.

        Spin-paired calculations: 1, spin-polarized calculation: 2."""
        if self.kpts is not None:
            nspin = set()
            for kpt in self.kpts:
                nspin.add(kpt.s)
            return len(nspin)
        return None

    def get_spin_polarized(self):
        """Is it a spin-polarized calculation?"""
        nos = self.get_number_of_spins()
        if nos is not None:
            return nos == 2
        return None

    def get_ibz_k_points(self):
        """Return k-points in the irreducible part of the Brillouin zone."""
        return self.ibz_kpts

    def get_kpt(self, kpt=0, spin=0):
        if self.kpts is not None:
            counter = 0
            for kpoint in self.kpts:
                if kpoint.s == spin:
                    if kpt == counter:
                        return kpoint
                    counter += 1
        return None

    def get_k_point_weights(self):
        """ Retunrs the weights of the k points """
        if self.kpts is not None:
            weights = []
            for kpoint in self.kpts:
                if kpoint.s == 0:
                    weights.append(kpoint.weight)
            return np.array(weights)
        return None

    def get_occupation_numbers(self, kpt=0, spin=0):
        """Return occupation number array."""
        kpoint = self.get_kpt(kpt, spin)
        if kpoint is not None:
            return kpoint.f_n
        return None

    def get_eigenvalues(self, kpt=0, spin=0):
        """Return eigenvalue array."""
        kpoint = self.get_kpt(kpt, spin)
        if kpoint is not None:
            return kpoint.eps_n
        return None

    def get_homo_lumo(self):
        """Return HOMO and LUMO energies."""
        if self.kpts is None:
            raise RuntimeError('No kpts')
        eHs = []
        eLs = []
        for kpt in self.kpts:
            eH, eL = self.get_homo_lumo_by_spin(kpt.s)
            eHs.append(eH)
            eLs.append(eL)
        return (np.array(eHs).max(), np.array(eLs).min())

    def get_homo_lumo_by_spin(self, spin=0):
        """Return HOMO and LUMO energies for a given spin."""
        if self.kpts is None:
            raise RuntimeError('No kpts')
        for kpt in self.kpts:
            if kpt.s == spin:
                break
        else:
            raise RuntimeError('No k-point with spin {0}'.format(spin))
        if self.eFermi is None:
            raise RuntimeError('Fermi level is not available')
        eH = -1e+32
        eL = 1e+32
        for kpt in self.kpts:
            if kpt.s == spin:
                for e in kpt.eps_n:
                    if e <= self.eFermi:
                        eH = max(eH, e)
                    else:
                        eL = min(eL, e)
        return (eH, eL)