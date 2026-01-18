from __future__ import annotations
import functools
import warnings
from collections import namedtuple
from typing import TYPE_CHECKING, NamedTuple
import numpy as np
from monty.json import MSONable
from scipy.constants import value as _cd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert
from pymatgen.core import Structure, get_el_sp
from pymatgen.core.spectrum import Spectrum
from pymatgen.electronic_structure.core import Orbital, OrbitalType, Spin
from pymatgen.util.coord import get_linear_interpolated_value
class CompleteDos(Dos):
    """This wrapper class defines a total dos, and also provides a list of PDos.
    Mainly used by pymatgen.io.vasp.Vasprun to create a complete Dos from
    a vasprun.xml file. You are unlikely to try to generate this object
    manually.

    Attributes:
        structure (Structure): Structure associated with the CompleteDos.
        pdos (dict): Dict of partial densities of the form {Site:{Orbital:{Spin:Densities}}}.
    """

    def __init__(self, structure: Structure, total_dos: Dos, pdoss: Mapping[PeriodicSite, Mapping[Orbital, Mapping[Spin, ArrayLike]]], normalize: bool=False) -> None:
        """
        Args:
            structure: Structure associated with this particular DOS.
            total_dos: total Dos for structure
            pdoss: The pdoss are supplied as an {Site: {Orbital: {Spin:Densities}}}
            normalize: Whether to normalize the densities by the volume of the structure.
                If True, the units of the densities are states/eV/Angstrom^3. Otherwise,
                the units are states/eV.
        """
        vol = structure.volume if normalize else None
        super().__init__(total_dos.efermi, energies=total_dos.energies, densities={k: np.array(d) for k, d in total_dos.densities.items()}, norm_vol=vol)
        self.pdos = pdoss
        self.structure = structure

    def get_normalized(self) -> CompleteDos:
        """Returns a normalized version of the CompleteDos."""
        if self.norm_vol is not None:
            return self
        return CompleteDos(structure=self.structure, total_dos=self, pdoss=self.pdos, normalize=True)

    def get_site_orbital_dos(self, site: PeriodicSite, orbital: Orbital) -> Dos:
        """Get the Dos for a particular orbital of a particular site.

        Args:
            site: Site in Structure associated with CompleteDos.
            orbital: Orbital in the site.

        Returns:
            Dos containing densities for orbital of site.
        """
        return Dos(self.efermi, self.energies, self.pdos[site][orbital])

    def get_site_dos(self, site: PeriodicSite) -> Dos:
        """Get the total Dos for a site (all orbitals).

        Args:
            site: Site in Structure associated with CompleteDos.

        Returns:
            Dos containing summed orbital densities for site.
        """
        site_dos = functools.reduce(add_densities, self.pdos[site].values())
        return Dos(self.efermi, self.energies, site_dos)

    def get_site_spd_dos(self, site: PeriodicSite) -> dict[OrbitalType, Dos]:
        """Get orbital projected Dos of a particular site.

        Args:
            site: Site in Structure associated with CompleteDos.

        Returns:
            dict of {OrbitalType: Dos}, e.g. {OrbitalType.s: Dos object, ...}
        """
        spd_dos: dict[OrbitalType, dict[Spin, np.ndarray]] = {}
        for orb, pdos in self.pdos[site].items():
            orbital_type = _get_orb_type(orb)
            if orbital_type in spd_dos:
                spd_dos[orbital_type] = add_densities(spd_dos[orbital_type], pdos)
            else:
                spd_dos[orbital_type] = pdos
        return {orb: Dos(self.efermi, self.energies, densities) for orb, densities in spd_dos.items()}

    def get_site_t2g_eg_resolved_dos(self, site: PeriodicSite) -> dict[str, Dos]:
        """Get the t2g, eg projected DOS for a particular site.

        Args:
            site: Site in Structure associated with CompleteDos.

        Returns:
            dict[str, Dos]: A dict {"e_g": Dos, "t2g": Dos} containing summed e_g and t2g DOS for the site.
        """
        t2g_dos = []
        eg_dos = []
        for s, atom_dos in self.pdos.items():
            if s == site:
                for orb, pdos in atom_dos.items():
                    if orb in (Orbital.dxy, Orbital.dxz, Orbital.dyz):
                        t2g_dos.append(pdos)
                    elif orb in (Orbital.dx2, Orbital.dz2):
                        eg_dos.append(pdos)
        return {'t2g': Dos(self.efermi, self.energies, functools.reduce(add_densities, t2g_dos)), 'e_g': Dos(self.efermi, self.energies, functools.reduce(add_densities, eg_dos))}

    def get_spd_dos(self) -> dict[OrbitalType, Dos]:
        """Get orbital projected Dos.

        Returns:
            dict[OrbitalType, Dos]: e.g. {OrbitalType.s: Dos object, ...}
        """
        spd_dos = {}
        for atom_dos in self.pdos.values():
            for orb, pdos in atom_dos.items():
                orbital_type = _get_orb_type(orb)
                if orbital_type not in spd_dos:
                    spd_dos[orbital_type] = pdos
                else:
                    spd_dos[orbital_type] = add_densities(spd_dos[orbital_type], pdos)
        return {orb: Dos(self.efermi, self.energies, densities) for orb, densities in spd_dos.items()}

    def get_element_dos(self) -> dict[SpeciesLike, Dos]:
        """Get element projected Dos.

        Returns:
            dict[Element, Dos]
        """
        el_dos = {}
        for site, atom_dos in self.pdos.items():
            el = site.specie
            for pdos in atom_dos.values():
                if el not in el_dos:
                    el_dos[el] = pdos
                else:
                    el_dos[el] = add_densities(el_dos[el], pdos)
        return {el: Dos(self.efermi, self.energies, densities) for el, densities in el_dos.items()}

    def get_element_spd_dos(self, el: SpeciesLike) -> dict[OrbitalType, Dos]:
        """Get element and spd projected Dos.

        Args:
            el: Element in Structure.composition associated with CompleteDos

        Returns:
            dict[OrbitalType, Dos]: e.g. {OrbitalType.s: Dos object, ...}
        """
        el = get_el_sp(el)
        el_dos = {}
        for site, atom_dos in self.pdos.items():
            if site.specie == el:
                for orb, pdos in atom_dos.items():
                    orbital_type = _get_orb_type(orb)
                    if orbital_type not in el_dos:
                        el_dos[orbital_type] = pdos
                    else:
                        el_dos[orbital_type] = add_densities(el_dos[orbital_type], pdos)
        return {orb: Dos(self.efermi, self.energies, densities) for orb, densities in el_dos.items()}

    @property
    def spin_polarization(self) -> float | None:
        """Calculates spin polarization at Fermi level. If the
        calculation is not spin-polarized, None will be returned.

        See Sanvito et al., doi: 10.1126/sciadv.1602241 for an example usage.

        Returns:
            float: spin polarization in range [0, 1], will also return NaN if spin
                polarization ill-defined (e.g. for insulator).
        """
        n_F = self.get_interpolated_value(self.efermi)
        n_F_up = n_F[Spin.up]
        if Spin.down not in n_F:
            return None
        n_F_down = n_F[Spin.down]
        if n_F_up + n_F_down == 0:
            return float('NaN')
        spin_polarization = (n_F_up - n_F_down) / (n_F_up + n_F_down)
        return abs(spin_polarization)

    def get_band_filling(self, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None, spin: Spin | None=None) -> float:
        """Compute the orbital-projected band filling, defined as the zeroth moment
        up to the Fermi level.

        Args:
            band: Orbital type to get the band center of (default is d-band)
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            spin: Spin channel to use. By default, the spin channels will be combined.

        Returns:
            float: band filling in eV, often denoted f_d for the d-band
        """
        if elements and sites:
            raise ValueError('Both element and site cannot be specified.')
        densities: dict[Spin, ArrayLike] = {}
        if elements:
            for idx, el in enumerate(elements):
                spd_dos = self.get_element_spd_dos(el)[band]
                densities = spd_dos.densities if idx == 0 else add_densities(densities, spd_dos.densities)
            dos = Dos(self.efermi, self.energies, densities)
        elif sites:
            for idx, site in enumerate(sites):
                spd_dos = self.get_site_spd_dos(site)[band]
                densities = spd_dos.densities if idx == 0 else add_densities(densities, spd_dos.densities)
            dos = Dos(self.efermi, self.energies, densities)
        else:
            dos = self.get_spd_dos()[band]
        energies = dos.energies - dos.efermi
        dos_densities = dos.get_densities(spin=spin)
        energies = dos.energies - dos.efermi
        return np.trapz(dos_densities[energies < 0], x=energies[energies < 0]) / np.trapz(dos_densities, x=energies)

    def get_band_center(self, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None, spin: Spin | None=None, erange: list[float] | None=None) -> float:
        """Compute the orbital-projected band center, defined as the first moment
        relative to the Fermi level
            int_{-inf}^{+inf} rho(E)*E dE/int_{-inf}^{+inf} rho(E) dE
        based on the work of Hammer and Norskov, Surf. Sci., 343 (1995) where the
        limits of the integration can be modified by erange and E is the set
        of energies taken with respect to the Fermi level. Note that the band center
        is often highly sensitive to the selected erange.

        Args:
            band: Orbital type to get the band center of (default is d-band)
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            spin: Spin channel to use. By default, the spin channels will be combined.
            erange: [min, max] energy range to consider, with respect to the Fermi level.
                Default is None, which means all energies are considered.

        Returns:
            float: band center in eV, often denoted epsilon_d for the d-band center
        """
        return self.get_n_moment(1, elements=elements, sites=sites, band=band, spin=spin, erange=erange, center=False)

    def get_band_width(self, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None, spin: Spin | None=None, erange: list[float] | None=None) -> float:
        """Get the orbital-projected band width, defined as the square root of the second moment
            sqrt(int_{-inf}^{+inf} rho(E)*(E-E_center)^2 dE/int_{-inf}^{+inf} rho(E) dE)
        where E_center is the orbital-projected band center, the limits of the integration can be
        modified by erange, and E is the set of energies taken with respect to the Fermi level.
        Note that the band width is often highly sensitive to the selected erange.

        Args:
            band: Orbital type to get the band center of (default is d-band)
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            spin: Spin channel to use. By default, the spin channels will be combined.
            erange: [min, max] energy range to consider, with respect to the Fermi level.
                Default is None, which means all energies are considered.

        Returns:
            float: Orbital-projected band width in eV
        """
        return np.sqrt(self.get_n_moment(2, elements=elements, sites=sites, band=band, spin=spin, erange=erange))

    def get_band_skewness(self, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None, spin: Spin | None=None, erange: list[float] | None=None) -> float:
        """Get the orbital-projected skewness, defined as the third standardized moment
            int_{-inf}^{+inf} rho(E)*(E-E_center)^3 dE/int_{-inf}^{+inf} rho(E) dE)
            /
            (int_{-inf}^{+inf} rho(E)*(E-E_center)^2 dE/int_{-inf}^{+inf} rho(E) dE))^(3/2)
        where E_center is the orbital-projected band center, the limits of the integration can be
        modified by erange, and E is the set of energies taken with respect to the Fermi level.
        Note that the skewness is often highly sensitive to the selected erange.

        Args:
            band: Orbitals to get the band center of (default is d-band)
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            spin: Spin channel to use. By default, the spin channels will be combined.
            erange: [min, max] energy range to consider, with respect to the Fermi level.
                Default is None, which means all energies are considered.

        Returns:
            float: orbital-projected skewness (dimensionless)
        """
        kwds: dict = dict(elements=elements, sites=sites, band=band, spin=spin, erange=erange)
        return self.get_n_moment(3, **kwds) / self.get_n_moment(2, **kwds) ** (3 / 2)

    def get_band_kurtosis(self, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None, spin: Spin | None=None, erange: list[float] | None=None) -> float:
        """Get the orbital-projected kurtosis, defined as the fourth standardized moment
            int_{-inf}^{+inf} rho(E)*(E-E_center)^4 dE/int_{-inf}^{+inf} rho(E) dE)
            /
            (int_{-inf}^{+inf} rho(E)*(E-E_center)^2 dE/int_{-inf}^{+inf} rho(E) dE))^2
        where E_center is the orbital-projected band center, the limits of the integration can be
        modified by erange, and E is the set of energies taken with respect to the Fermi level.
        Note that the skewness is often highly sensitive to the selected erange.

        Args:
            band: Orbital type to get the band center of (default is d-band)
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            spin: Spin channel to use. By default, the spin channels will be combined.
            erange: [min, max] energy range to consider, with respect to the Fermi level.
                Default is None, which means all energies are considered.

        Returns:
            float: orbital-projected kurtosis (dimensionless)
        """
        kwds: dict = dict(elements=elements, sites=sites, band=band, spin=spin, erange=erange)
        return self.get_n_moment(4, **kwds) / self.get_n_moment(2, **kwds) ** 2

    def get_n_moment(self, n: int, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None, spin: Spin | None=None, erange: list[float] | None=None, center: bool=True) -> float:
        """Get the nth moment of the DOS centered around the orbital-projected band center, defined as
            int_{-inf}^{+inf} rho(E)*(E-E_center)^n dE/int_{-inf}^{+inf} rho(E) dE
        where n is the order, E_center is the orbital-projected band center, the limits of the integration can be
        modified by erange, and E is the set of energies taken with respect to the Fermi level. If center is False,
        then the E_center reference is not used.

        Args:
            n: The order for the moment
            band: Orbital type to get the band center of (default is d-band)
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            spin: Spin channel to use. By default, the spin channels will be combined.
            erange: [min, max] energy range to consider, with respect to the Fermi level.
                Default is None, which means all energies are considered.
            center: Take moments with respect to the band center

        Returns:
            Orbital-projected nth moment in eV
        """
        if elements and sites:
            raise ValueError('Both element and site cannot be specified.')
        densities: Mapping[Spin, ArrayLike] = {}
        if elements:
            for idx, el in enumerate(elements):
                spd_dos = self.get_element_spd_dos(el)[band]
                densities = spd_dos.densities if idx == 0 else add_densities(densities, spd_dos.densities)
            dos = Dos(self.efermi, self.energies, densities)
        elif sites:
            for idx, site in enumerate(sites):
                spd_dos = self.get_site_spd_dos(site)[band]
                densities = spd_dos.densities if idx == 0 else add_densities(densities, spd_dos.densities)
            dos = Dos(self.efermi, self.energies, densities)
        else:
            dos = self.get_spd_dos()[band]
        energies = dos.energies - dos.efermi
        dos_densities = dos.get_densities(spin=spin)
        if erange:
            dos_densities = dos_densities[(energies >= erange[0]) & (energies <= erange[1])]
            energies = energies[(energies >= erange[0]) & (energies <= erange[1])]
        if center:
            band_center = self.get_band_center(elements=elements, sites=sites, band=band, spin=spin, erange=erange)
            p = energies - band_center
        else:
            p = energies
        return np.trapz(p ** n * dos_densities, x=energies) / np.trapz(dos_densities, x=energies)

    def get_hilbert_transform(self, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None) -> Dos:
        """Return the Hilbert transform of the orbital-projected density of states,
        often plotted for a Newns-Anderson analysis.

        Args:
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            band: Orbitals to get the band center of (default is d-band)

        Returns:
            Hilbert transformation of the projected DOS.
        """
        if elements and sites:
            raise ValueError('Both element and site cannot be specified.')
        densities: Mapping[Spin, ArrayLike] = {}
        if elements:
            for idx, el in enumerate(elements):
                spd_dos = self.get_element_spd_dos(el)[band]
                densities = spd_dos.densities if idx == 0 else add_densities(densities, spd_dos.densities)
            dos = Dos(self.efermi, self.energies, densities)
        elif sites:
            for idx, site in enumerate(sites):
                spd_dos = self.get_site_spd_dos(site)[band]
                densities = spd_dos.densities if idx == 0 else add_densities(densities, spd_dos.densities)
            dos = Dos(self.efermi, self.energies, densities)
        else:
            dos = self.get_spd_dos()[band]
        densities_transformed = {Spin.up: np.imag(hilbert(dos.get_densities(spin=Spin.up)))}
        if Spin.down in self.densities:
            densities_transformed[Spin.down] = np.imag(hilbert(dos.get_densities(spin=Spin.down)))
        return Dos(self.efermi, self.energies, densities_transformed)

    def get_upper_band_edge(self, band: OrbitalType=OrbitalType.d, elements: list[SpeciesLike] | None=None, sites: list[PeriodicSite] | None=None, spin: Spin | None=None, erange: list[float] | None=None) -> float:
        """Get the orbital-projected upper band edge. The definition by Xin et al.
        Phys. Rev. B, 89, 115114 (2014) is used, which is the highest peak position of the
        Hilbert transform of the orbital-projected DOS.

        Args:
            band: Orbital type to get the band center of (default is d-band)
            elements: Elements to get the band center of (cannot be used in conjunction with site)
            sites: Sites to get the band center of (cannot be used in conjunction with el)
            spin: Spin channel to use. By default, the spin channels will be combined.
            erange: [min, max] energy range to consider, with respect to the Fermi level.
                Default is None, which means all energies are considered.

        Returns:
            Upper band edge in eV, often denoted epsilon_u
        """
        transformed_dos = self.get_hilbert_transform(elements=elements, sites=sites, band=band)
        energies = transformed_dos.energies - transformed_dos.efermi
        densities = transformed_dos.get_densities(spin=spin)
        if erange:
            densities = densities[(energies >= erange[0]) & (energies <= erange[1])]
            energies = energies[(energies >= erange[0]) & (energies <= erange[1])]
        return energies[np.argmax(densities)]

    def get_dos_fp(self, type: str='summed_pdos', binning: bool=True, min_e: float | None=None, max_e: float | None=None, n_bins: int=256, normalize: bool=True) -> NamedTuple:
        """Generates the DOS fingerprint.

        Based on work of:

        F. Knoop, T. A. r Purcell, M. Scheffler, C. Carbogno, J. Open Source Softw. 2020, 5, 2671.
        Source - https://gitlab.com/vibes-developers/vibes/-/tree/master/vibes/materials_fp
        Copyright (c) 2020 Florian Knoop, Thomas A.R.Purcell, Matthias Scheffler, Christian Carbogno.

        Args:
            type (str): Specify fingerprint type needed can accept '{s/p/d/f/}summed_{pdos/tdos}'
            (default is summed_pdos)
            binning (bool): If true, the DOS fingerprint is binned using np.linspace and n_bins.
                Default is True.
            min_e (float): The minimum mode energy to include in the fingerprint (default is None)
            max_e (float): The maximum mode energy to include in the fingerprint (default is None)
            n_bins (int): Number of bins to be used in the fingerprint (default is 256)
            normalize (bool): If true, normalizes the area under fp to equal to 1. Default is True.

        Raises:
            ValueError: If type is not one of the accepted values {s/p/d/f/}summed_{pdos/tdos}.

        Returns:
            Fingerprint(namedtuple) : The electronic density of states fingerprint
                of format (energies, densities, type, n_bins)
        """
        fingerprint = namedtuple('fingerprint', 'energies densities type n_bins bin_width')
        energies = self.energies - self.efermi
        if max_e is None:
            max_e = np.max(energies)
        if min_e is None:
            min_e = np.min(energies)
        pdos_obj = self.get_spd_dos()
        pdos = {}
        for key in pdos_obj:
            dens = pdos_obj[key].get_densities()
            pdos[key.name] = dens
        pdos['summed_pdos'] = np.sum(list(pdos.values()), axis=0)
        pdos['tdos'] = self.get_densities()
        try:
            densities = pdos[type]
            if len(energies) < n_bins:
                inds = np.where((energies >= min_e) & (energies <= max_e))
                return fingerprint(energies[inds], densities[inds], type, len(energies), np.diff(energies)[0])
            if binning:
                ener_bounds = np.linspace(min_e, max_e, n_bins + 1)
                ener = ener_bounds[:-1] + (ener_bounds[1] - ener_bounds[0]) / 2.0
                bin_width = np.diff(ener)[0]
            else:
                ener_bounds = np.array(energies)
                ener = np.append(energies, [energies[-1] + np.abs(energies[-1]) / 10])
                n_bins = len(energies)
                bin_width = np.diff(energies)[0]
            dos_rebin = np.zeros(ener.shape)
            for ii, e1, e2 in zip(range(len(ener)), ener_bounds[0:-1], ener_bounds[1:]):
                inds = np.where((energies >= e1) & (energies < e2))
                dos_rebin[ii] = np.sum(densities[inds])
            if normalize:
                area = np.sum(dos_rebin * bin_width)
                dos_rebin_sc = dos_rebin / area
            else:
                dos_rebin_sc = dos_rebin
            return fingerprint(np.array([ener]), dos_rebin_sc, type, n_bins, bin_width)
        except KeyError:
            raise ValueError("Please recheck type requested, either the orbital projections unavailable in input DOS or there's a typo in type.")

    @staticmethod
    def fp_to_dict(fp: NamedTuple) -> dict:
        """Converts a fingerprint into a dictionary.

        Args:
            fp: The DOS fingerprint to be converted into a dictionary

        Returns:
            dict: A dict of the fingerprint Keys=type, Values=np.ndarray(energies, densities)
        """
        fp_dict = {}
        fp_dict[fp[2]] = np.array([fp[0], fp[1]], dtype='object').T
        return fp_dict

    @staticmethod
    def get_dos_fp_similarity(fp1: NamedTuple, fp2: NamedTuple, col: int=1, pt: int | str='All', normalize: bool=False, tanimoto: bool=False) -> float:
        """Calculates the similarity index (dot product) of two fingerprints.

        Args:
            fp1 (NamedTuple): The 1st dos fingerprint object
            fp2 (NamedTuple): The 2nd dos fingerprint object
            col (int): The item in the fingerprints (0:energies,1: densities) to take the dot product of (default is 1)
            pt (int or str) : The index of the point that the dot product is to be taken (default is All)
            normalize (bool): If True normalize the scalar product to 1 (default is False)
            tanimoto (bool): If True will compute Tanimoto index (default is False)

        Raises:
            ValueError: If both tanimoto and normalize are set to True.

        Returns:
            float: Similarity index given by the dot product
        """
        fp1_dict = CompleteDos.fp_to_dict(fp1) if not isinstance(fp1, dict) else fp1
        fp2_dict = CompleteDos.fp_to_dict(fp2) if not isinstance(fp2, dict) else fp2
        if pt == 'All':
            vec1 = np.array([pt[col] for pt in fp1_dict.values()]).flatten()
            vec2 = np.array([pt[col] for pt in fp2_dict.values()]).flatten()
        else:
            vec1 = fp1_dict[fp1[2][pt]][col]
            vec2 = fp2_dict[fp2[2][pt]][col]
        if not normalize and tanimoto:
            rescale = np.linalg.norm(vec1) ** 2 + np.linalg.norm(vec2) ** 2 - np.dot(vec1, vec2)
            return np.dot(vec1, vec2) / rescale
        if not tanimoto and normalize:
            rescale = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            return np.dot(vec1, vec2) / rescale
        if not tanimoto and (not normalize):
            rescale = 1.0
            return np.dot(vec1, vec2) / rescale
        raise ValueError('Cannot compute similarity index. Please set either normalize=True or tanimoto=True or both to False.')

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Returns CompleteDos object from dict representation."""
        tdos = Dos.from_dict(dct)
        struct = Structure.from_dict(dct['structure'])
        pdoss = {}
        for idx in range(len(dct['pdos'])):
            at = struct[idx]
            orb_dos = {}
            for orb_str, odos in dct['pdos'][idx].items():
                orb = Orbital[orb_str]
                orb_dos[orb] = {Spin(int(k)): v for k, v in odos['densities'].items()}
            pdoss[at] = orb_dos
        return cls(struct, tdos, pdoss)

    def as_dict(self) -> dict:
        """JSON-serializable dict representation of CompleteDos."""
        dct = {'@module': type(self).__module__, '@class': type(self).__name__, 'efermi': self.efermi, 'structure': self.structure.as_dict(), 'energies': self.energies.tolist(), 'densities': {str(spin): dens.tolist() for spin, dens in self.densities.items()}, 'pdos': []}
        if len(self.pdos) > 0:
            for at in self.structure:
                dd = {}
                for orb, pdos in self.pdos[at].items():
                    dd[str(orb)] = {'densities': {str(int(spin)): list(dens) for spin, dens in pdos.items()}}
                dct['pdos'].append(dd)
            dct['atom_dos'] = {str(at): dos.as_dict() for at, dos in self.get_element_dos().items()}
            dct['spd_dos'] = {str(orb): dos.as_dict() for orb, dos in self.get_spd_dos().items()}
        return dct

    def __str__(self) -> str:
        return f'Complete DOS for {self.structure}'