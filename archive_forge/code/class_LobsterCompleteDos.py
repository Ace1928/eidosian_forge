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
class LobsterCompleteDos(CompleteDos):
    """Extended CompleteDOS for Lobster."""

    def get_site_orbital_dos(self, site: PeriodicSite, orbital: str) -> Dos:
        """Get the Dos for a particular orbital of a particular site.

        Args:
            site: Site in Structure associated with CompleteDos.
            orbital: principal quantum number and orbital in string format, e.g. "4s".
                    possible orbitals are: "s", "p_y", "p_z", "p_x", "d_xy", "d_yz", "d_z^2",
                    "d_xz", "d_x^2-y^2", "f_y(3x^2-y^2)", "f_xyz",
                    "f_yz^2", "f_z^3", "f_xz^2", "f_z(x^2-y^2)", "f_x(x^2-3y^2)"
                    In contrast to the Cohpcar and the Cohplist objects, the strings from the Lobster files are used

        Returns:
            Dos containing densities of an orbital of a specific site.
        """
        if orbital[1:] not in ['s', 'p_y', 'p_z', 'p_x', 'd_xy', 'd_yz', 'd_z^2', 'd_xz', 'd_x^2-y^2', 'f_y(3x^2-y^2)', 'f_xyz', 'f_yz^2', 'f_z^3', 'f_xz^2', 'f_z(x^2-y^2)', 'f_x(x^2-3y^2)']:
            raise ValueError('orbital is not correct')
        return Dos(self.efermi, self.energies, self.pdos[site][orbital])

    def get_site_t2g_eg_resolved_dos(self, site: PeriodicSite) -> dict[str, Dos]:
        """Get the t2g, eg projected DOS for a particular site.

        Args:
            site: Site in Structure associated with CompleteDos.

        Returns:
            A dict {"e_g": Dos, "t2g": Dos} containing summed e_g and t2g DOS
            for the site.
        """
        warnings.warn('Are the orbitals correctly oriented? Are you sure?')
        t2g_dos = []
        eg_dos = []
        for s, atom_dos in self.pdos.items():
            if s == site:
                for orb, pdos in atom_dos.items():
                    if _get_orb_lobster(orb) in (Orbital.dxy, Orbital.dxz, Orbital.dyz):
                        t2g_dos.append(pdos)
                    elif _get_orb_lobster(orb) in (Orbital.dx2, Orbital.dz2):
                        eg_dos.append(pdos)
        return {'t2g': Dos(self.efermi, self.energies, functools.reduce(add_densities, t2g_dos)), 'e_g': Dos(self.efermi, self.energies, functools.reduce(add_densities, eg_dos))}

    def get_spd_dos(self) -> dict[str, Dos]:
        """Get orbital projected Dos.
        For example, if 3s and 4s are included in the basis of some element, they will be both summed in the orbital
        projected DOS.

        Returns:
            dict of {orbital: Dos}, e.g. {"s": Dos object, ...}
        """
        spd_dos = {}
        for atom_dos in self.pdos.values():
            for orb, pdos in atom_dos.items():
                orbital_type = _get_orb_type_lobster(orb)
                if orbital_type not in spd_dos:
                    spd_dos[orbital_type] = pdos
                else:
                    spd_dos[orbital_type] = add_densities(spd_dos[orbital_type], pdos)
        return {orb: Dos(self.efermi, self.energies, densities) for orb, densities in spd_dos.items()}

    def get_element_spd_dos(self, el: SpeciesLike) -> dict[str, Dos]:
        """Get element and spd projected Dos.

        Args:
            el: Element in Structure.composition associated with LobsterCompleteDos

        Returns:
            dict of {OrbitalType.s: densities, OrbitalType.p: densities, OrbitalType.d: densities}
        """
        el = get_el_sp(el)
        el_dos = {}
        for site, atom_dos in self.pdos.items():
            if site.specie == el:
                for orb, pdos in atom_dos.items():
                    orbital_type = _get_orb_type_lobster(orb)
                    if orbital_type not in el_dos:
                        el_dos[orbital_type] = pdos
                    else:
                        el_dos[orbital_type] = add_densities(el_dos[orbital_type], pdos)
        return {orb: Dos(self.efermi, self.energies, densities) for orb, densities in el_dos.items()}

    @classmethod
    def from_dict(cls, dct: dict) -> Self:
        """Hydrate CompleteDos object from dict representation."""
        tdos = Dos.from_dict(dct)
        struct = Structure.from_dict(dct['structure'])
        pdoss = {}
        for i in range(len(dct['pdos'])):
            at = struct[i]
            orb_dos = {}
            for orb_str, odos in dct['pdos'][i].items():
                orb = orb_str
                orb_dos[orb] = {Spin(int(k)): v for k, v in odos['densities'].items()}
            pdoss[at] = orb_dos
        return cls(struct, tdos, pdoss)