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
class VasprunBSLoader:
    """Loader for Bandstructure and Vasprun pmg objects."""

    def __init__(self, obj, structure=None, nelect=None) -> None:
        """
        Args:
            obj: Either a pmg Vasprun or a BandStructure object.
            structure: Structure object in case is not included in the BandStructure object.
            nelect: number of electrons in case a BandStructure obj is provided.

        Example:
            vrun = Vasprun('vasprun.xml')
            data = VasprunBSLoader(vrun)
        """
        if isinstance(obj, Vasprun):
            structure = obj.final_structure
            nelect = obj.parameters['NELECT']
            bs_obj = obj.get_band_structure()
        elif isinstance(obj, BandStructure):
            bs_obj = obj
        else:
            raise BoltztrapError('The object provided is neither a Bandstructure nor a Vasprun.')
        self.kpoints = np.array([kp.frac_coords for kp in bs_obj.kpoints])
        if bs_obj.structure:
            self.structure = bs_obj.structure
        elif structure:
            self.structure = structure
        else:
            raise BoltztrapError('A structure must be given.')
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
        self.mommat_all = None
        self.mommat = None
        self.magmom = None
        self.fermi = bs_obj.efermi * units.eV
        self.UCvol = self.structure.volume * units.Angstrom ** 3
        if not bs_obj.is_metal():
            self.vbm_idx = max(bs_obj.get_vbm()['band_index'][Spin.up] + bs_obj.get_vbm()['band_index'][Spin.down])
            self.cbm_idx = min(bs_obj.get_cbm()['band_index'][Spin.up] + bs_obj.get_cbm()['band_index'][Spin.down])
            self.vbm = bs_obj.get_vbm()['energy']
            self.cbm = bs_obj.get_cbm()['energy']
        else:
            self.vbm_idx = self.cbm_idx = None
            self.vbm = self.fermi / units.eV
            self.cbm = self.fermi / units.eV
        if nelect:
            self.nelect_all = nelect
        elif self.vbm_idx:
            self.nelect_all = self.vbm_idx + self.cbm_idx + 1
        else:
            raise BoltztrapError('nelect must be given.')

    @classmethod
    def from_file(cls, vasprun_file: str | Path) -> Self:
        """Get a vasprun.xml file and return a VasprunBSLoader."""
        vrun_obj = Vasprun(vasprun_file, parse_projected_eigen=True)
        return cls(vrun_obj)

    def get_lattvec(self):
        """The lattice vectors."""
        try:
            return self.lattvec
        except AttributeError:
            self.lattvec = self.atoms.get_cell().T * units.Angstrom
        return self.lattvec

    def get_volume(self):
        """Volume."""
        try:
            return self.UCvol
        except AttributeError:
            lattvec = self.get_lattvec()
            self.UCvol = np.abs(np.linalg.det(lattvec))
        return self.UCvol

    def bandana(self, emin=-np.inf, emax=np.inf):
        """Cut out bands outside the range (emin,emax)."""
        bandmin = np.min(self.ebands_all, axis=1)
        bandmax = np.max(self.ebands_all, axis=1)
        ntoolow = np.count_nonzero(bandmax <= emin)
        accepted = np.logical_and(bandmin < emax, bandmax > emin)
        self.ebands = self.ebands_all[accepted]
        self.proj = {}
        if self.proj_all:
            if len(self.proj_all) == 2:
                h = len(accepted) // 2
                self.proj[Spin.up] = self.proj_all[Spin.up][:, accepted[:h], :, :]
                self.proj[Spin.down] = self.proj_all[Spin.down][:, accepted[h:], :, :]
            elif len(self.proj_all) == 1:
                self.proj[Spin.up] = self.proj_all[Spin.up][:, accepted, :, :]
        if self.mommat_all:
            self.mommat = self.mommat[:, accepted, :]
        if self.nelect_all:
            self.nelect = self.nelect_all - self.dosweight * ntoolow
        return accepted