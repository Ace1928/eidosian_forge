from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import matplotlib.pyplot as plt
import numpy as np
from monty.serialization import dumpfn, loadfn
from tqdm import tqdm
from pymatgen.electronic_structure.bandstructure import BandStructure, BandStructureSymmLine, Spin
from pymatgen.electronic_structure.boltztrap import BoltztrapError
from pymatgen.electronic_structure.dos import CompleteDos, Dos, Orbital
from pymatgen.electronic_structure.plotter import BSPlotter, DosPlotter
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Vasprun
from pymatgen.symmetry.bandstructure import HighSymmKpath
class BandstructureLoader:
    """Loader for Bandstructure object."""

    def __init__(self, bs_obj, structure=None, nelect=None, mommat=None, magmom=None) -> None:
        """
        Args:
            bs_obj: BandStructure object.
            structure: Structure object. It is needed if it is not contained in the BandStructure obj.
            nelect: Number of electrons in the calculation.
            mommat: Matrix of derivatives of energy eigenvalues. TODO Not implemented yet.
            magmom: Matrix of magnetic moments in non collinear calculations. Not implemented yet.

        Example:
            vrun = Vasprun('vasprun.xml')
            bs = vrun.get_band_structure()
            st = vrun.final_structure
            ne = vrun.parameters['NELECT']
            data = BandstructureLoader(bs,st,ne)
        """
        warnings.warn('Deprecated Loader. Use VasprunBSLoader instead.')
        self.kpoints = np.array([kp.frac_coords for kp in bs_obj.kpoints])
        self.structure = bs_obj.structure if structure is None else structure
        self.atoms = AseAtomsAdaptor.get_atoms(self.structure)
        self.proj_all = None
        if bs_obj.projections:
            self.proj_all = {sp: p.transpose((1, 0, 3, 2)) for sp, p in bs_obj.projections.items()}
        e = np.array(list(bs_obj.bands.values()))
        e = e.reshape(-1, e.shape[-1])
        self.ebands_all = e * units.eV
        self.is_spin_polarized = bs_obj.is_spin_polarized
        self.dosweight = 1.0 if bs_obj.is_spin_polarized else 2.0
        self.lattvec = self.atoms.get_cell().T * units.Angstrom
        self.mommat_all = mommat
        self.mommat = mommat
        self.magmom = magmom
        self.fermi = bs_obj.efermi * units.eV
        self.UCvol = self.structure.volume * units.Angstrom ** 3
        if not bs_obj.is_metal():
            self.vbm_idx = max(bs_obj.get_vbm()['band_index'][Spin.up] + bs_obj.get_vbm()['band_index'][Spin.down])
            self.cbm_idx = min(bs_obj.get_cbm()['band_index'][Spin.up] + bs_obj.get_cbm()['band_index'][Spin.down])
            self.vbm = bs_obj.get_vbm()['energy']
            self.cbm = bs_obj.get_cbm()['energy']
            self.nelect_all = self.vbm_idx * self.dosweight
        else:
            self.vbm_idx = self.cbm_idx = None
            self.vbm = self.fermi
            self.cbm = self.fermi
            self.nelect_all = nelect

    def get_lattvec(self):
        """The lattice vectors."""
        try:
            return self.lattvec
        except AttributeError:
            self.lattvec = self.atoms.get_cell().T * units.Angstrom
        return self.lattvec

    def bandana(self, emin=-np.inf, emax=np.inf):
        """Cut out bands outside the range (emin,emax)."""
        band_min = np.min(self.ebands_all, axis=1)
        band_max = np.max(self.ebands_all, axis=1)
        n_too_low = np.count_nonzero(band_max <= emin)
        accepted = np.logical_and(band_min < emax, band_max > emin)
        self.ebands = self.ebands_all[accepted]
        self.proj = {}
        if self.proj_all:
            if len(self.proj_all) == 2:
                h = len(accepted) // 2
                self.proj[Spin.up] = self.proj_all[Spin.up][:, accepted[:h], :, :]
                self.proj[Spin.down] = self.proj_all[Spin.down][:, accepted[h:], :, :]
            elif len(self.proj) == 1:
                self.proj[Spin.up] = self.proj_all[Spin.up][:, accepted, :, :]
        if self.mommat_all:
            self.mommat = self.mommat[:, accepted, :]
        if self.nelect_all:
            self.nelect = self.nelect_all - self.dosweight * n_too_low
        return accepted

    def set_upper_lower_bands(self, e_lower, e_upper) -> None:
        """Set fake upper/lower bands, useful to set the same energy
        range in the spin up/down bands when calculating the DOS.
        """
        warnings.warn('This method does not work anymore in case of spin polarized case due to the concatenation of bands !')
        lower_band = e_lower * np.ones((1, self.ebands.shape[1]))
        upper_band = e_upper * np.ones((1, self.ebands.shape[1]))
        self.ebands = np.vstack((lower_band, self.ebands, upper_band))
        if self.proj:
            for sp, proj in self.proj.items():
                proj_lower = proj[:, 0:1, :, :]
                proj_upper = proj[:, -1:, :, :]
                self.proj[sp] = np.concatenate((proj_lower, proj, proj_upper), axis=1)

    def get_volume(self):
        """Volume."""
        try:
            return self.UCvol
        except AttributeError:
            lattvec = self.get_lattvec()
            self.UCvol = np.abs(np.linalg.det(lattvec))
        return self.UCvol