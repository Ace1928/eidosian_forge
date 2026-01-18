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
class VasprunLoader:
    """Loader for Vasprun object."""

    def __init__(self, vrun_obj=None) -> None:
        """vrun_obj: Vasprun object."""
        warnings.warn('Deprecated Loader. Use VasprunBSLoader instead.')
        if vrun_obj:
            self.kpoints = np.array(vrun_obj.actual_kpoints)
            self.structure = vrun_obj.final_structure
            self.atoms = AseAtomsAdaptor.get_atoms(self.structure)
            self.proj = None
            if len(vrun_obj.eigenvalues) == 1:
                e = next(iter(vrun_obj.eigenvalues.values()))
                self.ebands = e[:, :, 0].transpose() * units.eV
                self.dosweight = 2.0
                if vrun_obj.projected_eigenvalues:
                    self.proj = next(iter(vrun_obj.projected_eigenvalues.values()))
            elif len(vrun_obj.eigenvalues) == 2:
                raise BoltztrapError('spin bs case not implemented')
            self.lattvec = self.atoms.get_cell().T * units.Angstrom
            self.mommat = self.magmom = self.spin = None
            self.fermi = vrun_obj.efermi * units.eV
            self.nelect = vrun_obj.parameters['NELECT']
            self.UCvol = self.structure.volume * units.Angstrom ** 3
            bs_obj = vrun_obj.get_band_structure()
            if not bs_obj.is_metal():
                self.vbm_idx = max(bs_obj.get_vbm()['band_index'][Spin.up] + bs_obj.get_vbm()['band_index'][Spin.down])
                self.cbm_idx = min(bs_obj.get_cbm()['band_index'][Spin.up] + bs_obj.get_cbm()['band_index'][Spin.down])
                self.vbm = bs_obj.get_vbm()['energy']
                self.cbm = bs_obj.get_cbm()['energy']
            else:
                self.vbm_idx = self.cbm_idx = None
                self.vbm = self.fermi
                self.cbm = self.fermi

    @classmethod
    def from_file(cls, vasprun_file: str | Path) -> Self:
        """Get a vasprun.xml file and return a VasprunLoader."""
        vrun_obj = Vasprun(vasprun_file, parse_projected_eigen=True)
        return cls(vrun_obj)

    def get_lattvec(self):
        """Lattice vectors."""
        try:
            return self.lattvec
        except AttributeError:
            self.lattvec = self.atoms.get_cell().T * units.Angstrom
        return self.lattvec

    def bandana(self, emin=-np.inf, emax=np.inf):
        """Cut out bands outside the range (emin,emax)."""
        band_min = np.min(self.ebands, axis=1)
        band_max = np.max(self.ebands, axis=1)
        ii = np.nonzero(band_min < emax)
        n_emax = ii[0][-1]
        ii = np.nonzero(band_max > emin)
        n_emin = ii[0][0]
        self.ebands = self.ebands[n_emin:n_emax + 1]
        if isinstance(self.proj, np.ndarray):
            self.proj = self.proj[:, n_emin:n_emax + 1, :, :]
        if self.mommat is not None:
            self.mommat = self.mommat[:, n_emin:n_emax + 1, :]
        if self.nelect is not None:
            self.nelect -= self.dosweight * n_emin
        return (n_emin, n_emax)

    def get_volume(self):
        """Volume of cell."""
        try:
            return self.UCvol
        except AttributeError:
            latt_vec = self.get_lattvec()
            self.UCvol = np.abs(np.linalg.det(latt_vec))
        return self.UCvol