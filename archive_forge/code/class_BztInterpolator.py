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
class BztInterpolator:
    """Interpolate the dft band structures."""

    def __init__(self, data, lpfac=10, energy_range=1.5, curvature=True, save_bztInterp=False, load_bztInterp=False, save_bands=False, fname='bztInterp.json.gz') -> None:
        """
        Args:
            data: A loader
            lpfac: the number of interpolation points in the real space. By
                default 10 gives 10 time more points in the real space than
                the number of kpoints given in reciprocal space.
            energy_range: usually the interpolation is not needed on the entire energy
                range but on a specific range around the Fermi level.
                This energy in eV fix the range around the Fermi level
                (E_fermi-energy_range,E_fermi+energy_range) of
                bands that will be interpolated
                and taken into account to calculate the transport properties.
            curvature: boolean value to enable/disable the calculation of second
                derivative related transport properties (Hall coefficient).
            save_bztInterp: Default False. If True coefficients and equivalences are
                saved in fname file.
            load_bztInterp: Default False. If True the coefficients and equivalences
                are loaded from fname file, not calculated. It can be faster than
                re-calculate them in some cases.
            save_bands: Default False. If True interpolated bands are also stored.
                It can be slower than interpolate them. Not recommended.
            fname: File path where to store/load from the coefficients and equivalences.

        Example:
            data = VasprunLoader().from_file('vasprun.xml')
            bztInterp = BztInterpolator(data)
        """
        bands_loaded = False
        self.data = data
        num_kpts = self.data.kpoints.shape[0]
        self.efermi = self.data.fermi
        middle_gap_en = (self.data.cbm + self.data.vbm) / 2
        self.accepted = self.data.bandana(emin=(middle_gap_en - energy_range) * units.eV, emax=(middle_gap_en + energy_range) * units.eV)
        if load_bztInterp:
            bands_loaded = self.load(fname)
        else:
            self.equivalences = sphere.get_equivalences(self.data.atoms, self.data.magmom, num_kpts * lpfac)
            self.coeffs = fite.fitde3D(self.data, self.equivalences)
        if not bands_loaded:
            self.eband, self.vvband, self.cband = fite.getBTPbands(self.equivalences, self.coeffs, self.data.lattvec, curvature=curvature)
        if save_bztInterp:
            self.save(fname, save_bands)

    def load(self, fname='bztInterp.json.gz'):
        """Load the coefficient, equivalences, bands from fname."""
        dct = loadfn(fname)
        if len(dct) > 2:
            self.equivalences, coeffs, self.eband, self.vvband, self.cband = dct
            bands_loaded = True
        elif len(dct) == 2:
            self.equivalences, coeffs = loadfn(fname)
            bands_loaded = False
        else:
            raise BoltztrapError('Something wrong reading the data file!')
        self.coeffs = coeffs[0] + coeffs[1] * 1j
        return bands_loaded

    def save(self, fname='bztInterp.json.gz', bands=False) -> None:
        """Save the coefficient, equivalences to fname.
        If bands is True, also interpolated bands are stored.
        """
        if bands:
            dumpfn([self.equivalences, [self.coeffs.real, self.coeffs.imag], self.eband, self.vvband, self.cband], fname)
        else:
            dumpfn([self.equivalences, [self.coeffs.real, self.coeffs.imag]], fname)

    def get_band_structure(self, kpaths=None, kpoints_lbls_dict=None, density=20):
        """Return a BandStructureSymmLine object interpolating bands along a
        High symmetry path calculated from the structure using HighSymmKpath
        function. If kpaths and kpoints_lbls_dict are provided, a custom
        path is interpolated.
        kpaths: List of lists of following kpoints labels defining
                the segments of the path. E.g. [['L','M'],['L','X']]
        kpoints_lbls_dict: Dict where keys are the kpoint labels used in kpaths
                and values are their fractional coordinates.
                E.g. {'L':np.array(0.5,0.5,0.5)},
                      'M':np.array(0.5,0.,0.5),
                      'X':np.array(0.5,0.5,0.)}
        density: Number of points in each segment.
        """
        if isinstance(kpaths, list) and isinstance(kpoints_lbls_dict, dict):
            kpoints = []
            for kpath in kpaths:
                for idx, k_pt in enumerate(kpath[:-1], start=1):
                    sta = kpoints_lbls_dict[k_pt]
                    end = kpoints_lbls_dict[kpath[idx]]
                    kpoints.append(np.linspace(sta, end, density))
            kpoints = np.concatenate(kpoints)
        else:
            kpath = HighSymmKpath(self.data.structure)
            kpoints = np.vstack(kpath.get_kpoints(density, coords_are_cartesian=False)[0])
            kpoints_lbls_dict = kpath.kpath['kpoints']
        lattvec = self.data.get_lattvec()
        egrid, _vgrid = fite.getBands(kpoints, self.equivalences, lattvec, self.coeffs)
        if self.data.is_spin_polarized:
            h = sum(np.array_split(self.accepted, 2)[0])
            egrid = np.array_split(egrid, [h], axis=0)
            bands_dict = {Spin.up: egrid[0] / units.eV, Spin.down: egrid[1] / units.eV}
        else:
            bands_dict = {Spin.up: egrid / units.eV}
        return BandStructureSymmLine(kpoints, bands_dict, self.data.structure.lattice.reciprocal_lattice, self.efermi / units.eV, labels_dict=kpoints_lbls_dict)

    def get_dos(self, partial_dos=False, npts_mu=10000, T=None, progress=False):
        """Return a Dos object interpolating bands.

        Args:
            partial_dos: if True, projections will be interpolated as well
                and partial doses will be return. Projections must be available
                in the loader.
            npts_mu: number of energy points of the Dos
            T: parameter used to smooth the Dos
            progress: Default False, If True a progress bar is shown when
                partial dos are computed.
        """
        dos_dict = {}
        enr = (self.eband.min(), self.eband.max())
        if self.data.is_spin_polarized:
            h = sum(np.array_split(self.accepted, 2)[0])
            eband_ud = np.array_split(self.eband, [h], axis=0)
            vvband_ud = np.array_split(self.vvband, [h], axis=0)
            spins = [Spin.up, Spin.down]
        else:
            eband_ud = [self.eband]
            vvband_ud = [self.vvband]
            spins = [Spin.up]
        for spin, eb, vvb in zip(spins, eband_ud, vvband_ud):
            energies, densities, _vvdos, _cdos = BL.BTPDOS(eb, vvb, npts=npts_mu, erange=enr)
            if T:
                densities = BL.smoothen_DOS(energies, densities, T)
            dos_dict.setdefault(spin, densities)
        tdos = Dos(self.efermi / units.eV, energies / units.eV, dos_dict)
        if partial_dos:
            tdos = self.get_partial_doses(tdos, eband_ud, spins, enr, npts_mu, T, progress)
        return tdos

    def get_partial_doses(self, tdos, eband_ud, spins, enr, npts_mu, T, progress):
        """Return a CompleteDos object interpolating the projections.

        tdos: total dos previously calculated
        npts_mu: number of energy points of the Dos
        T: parameter used to smooth the Dos
        progress: Default False, If True a progress bar is shown.
        """
        if not self.data.proj:
            raise BoltztrapError('No projections loaded.')
        bkp_data_ebands = np.copy(self.data.ebands)
        pdoss = {}
        if progress:
            n_iter = np.prod(np.sum([np.array(i.shape)[2:] for i in self.data.proj.values()]))
            t = tqdm(total=n_iter * 2)
        for spin, eb in zip(spins, eband_ud):
            for idx, site in enumerate(self.data.structure):
                if site not in pdoss:
                    pdoss[site] = {}
                for iorb, orb in enumerate(Orbital):
                    if progress:
                        t.update()
                    if iorb == self.data.proj[spin].shape[-1]:
                        break
                    if orb not in pdoss[site]:
                        pdoss[site][orb] = {}
                    self.data.ebands = self.data.proj[spin][:, :, idx, iorb].T
                    coeffs = fite.fitde3D(self.data, self.equivalences)
                    proj, _vvproj, _cproj = fite.getBTPbands(self.equivalences, coeffs, self.data.lattvec)
                    edos, pdos = BL.DOS(eb, npts=npts_mu, weights=np.abs(proj.real), erange=enr)
                    if T:
                        pdos = BL.smoothen_DOS(edos, pdos, T)
                    pdoss[site][orb][spin] = pdos
        self.data.ebands = bkp_data_ebands
        return CompleteDos(self.data.structure, total_dos=tdos, pdoss=pdoss)