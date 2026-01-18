from __future__ import annotations
import copy
import itertools
import json
import logging
import math
import os
import warnings
from functools import reduce
from math import gcd, isclose
from typing import TYPE_CHECKING, cast
import numpy as np
from monty.fractions import lcm
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, PeriodicSite, Structure, get_el_sp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.util.coord import in_coord_list
from pymatgen.util.due import Doi, due
class Slab(Structure):
    """Class to hold information for a Slab, with additional
    attributes pertaining to slabs, but the init method does not
    actually create a slab. Also has additional methods that returns other information
    about a Slab such as the surface area, normal, and atom adsorption.

    Note that all Slabs have the surface normal oriented perpendicular to the a
    and b lattice vectors. This means the lattice vectors a and b are in the
    surface plane and the c vector is out of the surface plane (though not
    necessarily perpendicular to the surface).
    """

    def __init__(self, lattice: Lattice | np.ndarray, species: Sequence[Any], coords: np.ndarray, miller_index: tuple[int, int, int], oriented_unit_cell: Structure, shift: float, scale_factor: np.ndarray, reorient_lattice: bool=True, validate_proximity: bool=False, to_unit_cell: bool=False, reconstruction: str | None=None, coords_are_cartesian: bool=False, site_properties: dict | None=None, energy: float | None=None) -> None:
        """A Structure object with additional information
        and methods pertaining to Slabs.

        Args:
            lattice (Lattice/3x3 array): The lattice, either as a
                pymatgen.core.Lattice or simply as any 2D array.
                Each row should correspond to a lattice
                vector. E.g., [[10,0,0], [20,10,0], [0,0,30]].
            species ([Species]): Sequence of species on each site. Can take in
                flexible input, including:

                i.  A sequence of element / species specified either as string
                    symbols, e.g. ["Li", "Fe2+", "P", ...] or atomic numbers,
                    e.g., (3, 56, ...) or actual Element or Species objects.

                ii. List of dict of elements/species and occupancies, e.g.,
                    [{"Fe": 0.5, "Mn": 0.5}, ...]. This allows the setup of
                    disordered structures.
            coords (Nx3 array): list of fractional/cartesian coordinates of each species.
            miller_index (tuple[h, k, l]): Miller index of plane parallel to
                surface. Note that this is referenced to the input structure. If
                you need this to be based on the conventional cell,
                you should supply the conventional structure.
            oriented_unit_cell (Structure): The oriented_unit_cell from which
                this Slab is created (by scaling in the c-direction).
            shift (float): The shift in the c-direction applied to get the
                termination.
            scale_factor (np.ndarray): scale_factor Final computed scale factor
                that brings the parent cell to the surface cell.
            reorient_lattice (bool): reorients the lattice parameters such that
                the c direction is along the z axis.
            validate_proximity (bool): Whether to check if there are sites
                that are less than 0.01 Ang apart. Defaults to False.
            reconstruction (str): Type of reconstruction. Defaults to None if
                the slab is not reconstructed.
            to_unit_cell (bool): Translates fractional coordinates into the unit cell. Defaults to False.
            coords_are_cartesian (bool): Set to True if you are providing
                coordinates in Cartesian coordinates. Defaults to False.
            site_properties (dict): Properties associated with the sites as a
                dict of sequences, e.g., {"magmom":[5,5,5,5]}. The sequences
                have to be the same length as the atomic species and
                fractional_coords. Defaults to None for no properties.
            energy (float): A value for the energy.
        """
        self.oriented_unit_cell = oriented_unit_cell
        self.miller_index = miller_index
        self.shift = shift
        self.reconstruction = reconstruction
        self.scale_factor = scale_factor
        self.energy = energy
        self.reorient_lattice = reorient_lattice
        if self.reorient_lattice:
            if coords_are_cartesian:
                coords = lattice.get_fractional_coords(coords)
                coords_are_cartesian = False
            lattice = Lattice.from_parameters(lattice.a, lattice.b, lattice.c, lattice.alpha, lattice.beta, lattice.gamma)
        super().__init__(lattice, species, coords, validate_proximity=validate_proximity, to_unit_cell=to_unit_cell, coords_are_cartesian=coords_are_cartesian, site_properties=site_properties)

    def __str__(self) -> str:
        outs = [f'Slab Summary ({self.composition.formula})', f'Reduced Formula: {self.composition.reduced_formula}', f'Miller index: {self.miller_index}', f'Shift: {self.shift:.4f}, Scale Factor: {self.scale_factor}', f'abc   : {' '.join((f'{i:0.6f}'.rjust(10) for i in self.lattice.abc))}', f'angles: {' '.join((f'{i:0.6f}'.rjust(10) for i in self.lattice.angles))}', f'Sites ({len(self)})']
        for idx, site in enumerate(self):
            outs.append(f'{idx + 1} {site.species_string} {' '.join((f'{j:0.6f}'.rjust(12) for j in site.frac_coords))}')
        return '\n'.join(outs)

    @property
    def center_of_mass(self) -> np.ndarray:
        """The center of mass of the Slab in fractional coordinates."""
        weights = [site.species.weight for site in self]
        return np.average(self.frac_coords, weights=weights, axis=0)

    @property
    def dipole(self) -> np.ndarray:
        """The dipole moment of the Slab in the direction of the surface normal.

        Note that the Slab must be oxidation state decorated for this to work properly.
        Otherwise, the Slab will always have a dipole moment of 0.
        """
        centroid = np.sum(self.cart_coords, axis=0) / len(self)
        dipole = np.zeros(3)
        for site in self:
            charge = sum((getattr(sp, 'oxi_state', 0) * amt for sp, amt in site.species.items()))
            dipole += charge * np.dot(site.coords - centroid, self.normal) * self.normal
        return dipole

    @property
    def normal(self) -> np.ndarray:
        """The surface normal vector of the Slab, normalized to unit length."""
        normal = np.cross(self.lattice.matrix[0], self.lattice.matrix[1])
        normal /= np.linalg.norm(normal)
        return normal

    @property
    def surface_area(self) -> float:
        """The surface area of the Slab."""
        matrix = self.lattice.matrix
        return np.linalg.norm(np.cross(matrix[0], matrix[1]))

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> Self:
        """
        Args:
            dct: dict.

        Returns:
            Creates slab from dict.
        """
        lattice = Lattice.from_dict(dct['lattice'])
        sites = [PeriodicSite.from_dict(sd, lattice) for sd in dct['sites']]
        struct = Structure.from_sites(sites)
        return Slab(lattice=lattice, species=struct.species_and_occu, coords=struct.frac_coords, miller_index=dct['miller_index'], oriented_unit_cell=Structure.from_dict(dct['oriented_unit_cell']), shift=dct['shift'], scale_factor=np.array(dct['scale_factor']), site_properties=struct.site_properties, energy=dct['energy'])

    def as_dict(self, **kwargs) -> dict:
        """MSONable dict."""
        dct = super().as_dict(**kwargs)
        dct['@module'] = type(self).__module__
        dct['@class'] = type(self).__name__
        dct['oriented_unit_cell'] = self.oriented_unit_cell.as_dict()
        dct['miller_index'] = self.miller_index
        dct['shift'] = self.shift
        dct['scale_factor'] = self.scale_factor.tolist()
        dct['reconstruction'] = self.reconstruction
        dct['energy'] = self.energy
        return dct

    def copy(self, site_properties: dict[str, Any] | None=None) -> Slab:
        """Get a copy of the structure, with options to update site properties.

        Args:
            site_properties (dict): Properties to update. The
                properties are specified in the same way as the constructor,
                i.e., as a dict of the form {property: [values]}.

        Returns:
            A copy of the Structure, with optionally new site_properties
        """
        props = self.site_properties
        if site_properties:
            props.update(site_properties)
        return Slab(self.lattice, self.species_and_occu, self.frac_coords, self.miller_index, self.oriented_unit_cell, self.shift, self.scale_factor, site_properties=props, reorient_lattice=self.reorient_lattice)

    def is_symmetric(self, symprec: float=0.1) -> bool:
        """Check if Slab is symmetric, i.e., contains inversion, mirror on (hkl) plane,
            or screw axis (rotation and translation) about [hkl].

        Args:
            symprec (float): Symmetry precision used for SpaceGroup analyzer.

        Returns:
            bool: Whether surfaces are symmetric.
        """
        spg_analyzer = SpacegroupAnalyzer(self, symprec=symprec)
        symm_ops = spg_analyzer.get_point_group_operations()
        return spg_analyzer.is_laue() or any((op.translation_vector[2] != 0 for op in symm_ops)) or any((np.all(op.rotation_matrix[2] == np.array([0, 0, -1])) for op in symm_ops))

    def is_polar(self, tol_dipole_per_unit_area: float=0.001) -> bool:
        """Check if the Slab is polar by computing the normalized dipole per unit area.
        Normalized dipole per unit area is used as it is more reliable than
        using the absolute value, which varies with surface area.

        Note that the Slab must be oxidation state decorated for this to work properly.
        Otherwise, the Slab will always have a dipole moment of 0.

        Args:
            tol_dipole_per_unit_area (float): A tolerance above which the Slab is
                considered polar.
        """
        dip_per_unit_area = self.dipole / self.surface_area
        return np.linalg.norm(dip_per_unit_area) > tol_dipole_per_unit_area

    def get_surface_sites(self, tag: bool=False) -> dict[str, list]:
        """Returns the surface sites and their indices in a dictionary.
        Useful for analysis involving broken bonds and for finding adsorption sites.

        The oriented unit cell of the slab will determine the
        coordination number of a typical site.
        We use VoronoiNN to determine the coordination number of sites.
        Due to the pathological error resulting from some surface sites in the
        VoronoiNN, we assume any site that has this error is a surface
        site as well. This will only work for single-element systems for now.

        Args:
            tag (bool): Option to adds site attribute "is_surfsite" (bool)
                to all sites of slab. Defaults to False

        Returns:
            A dictionary grouping sites on top and bottom of the slab together.
                {"top": [sites with indices], "bottom": [sites with indices]}

        Todo:
            Is there a way to determine site equivalence between sites in a slab
            and bulk system? This would allow us get the coordination number of
            a specific site for multi-elemental systems or systems with more
            than one inequivalent site. This will allow us to use this for
            compound systems.
        """
        from pymatgen.analysis.local_env import VoronoiNN
        spg_analyzer = SpacegroupAnalyzer(self.oriented_unit_cell)
        u_cell = spg_analyzer.get_symmetrized_structure()
        cn_dict: dict = {}
        voronoi_nn = VoronoiNN()
        unique_indices = [equ[0] for equ in u_cell.equivalent_indices]
        for idx in unique_indices:
            el = u_cell[idx].species_string
            if el not in cn_dict:
                cn_dict[el] = []
            cn = voronoi_nn.get_cn(u_cell, idx, use_weights=True)
            cn = float(f'{round(cn, 5):.5f}')
            if cn not in cn_dict[el]:
                cn_dict[el].append(cn)
        voronoi_nn = VoronoiNN()
        surf_sites_dict: dict = {'top': [], 'bottom': []}
        properties: list = []
        for idx, site in enumerate(self):
            is_top: bool = site.frac_coords[2] > self.center_of_mass[2]
            try:
                cn = float(f'{round(voronoi_nn.get_cn(self, idx, use_weights=True), 5):.5f}')
                if cn < min(cn_dict[site.species_string]):
                    properties.append(True)
                    key = 'top' if is_top else 'bottom'
                    surf_sites_dict[key].append([site, idx])
                else:
                    properties.append(False)
            except RuntimeError:
                properties.append(True)
                key = 'top' if is_top else 'bottom'
                surf_sites_dict[key].append([site, idx])
        if tag:
            self.add_site_property('is_surf_site', properties)
        return surf_sites_dict

    def get_symmetric_site(self, point: ArrayLike, cartesian: bool=False) -> ArrayLike:
        """This method uses symmetry operations to find an equivalent site on
        the other side of the slab. Works mainly for slabs with Laue symmetry.

        This is useful for retaining the non-polar and
        symmetric properties of a slab when creating adsorbed
        structures or symmetric reconstructions.

        TODO (@DanielYang59): use "site" over "point" as arg name for consistency

        Args:
            point (ArrayLike): Fractional coordinate of the original site.
            cartesian (bool): Use Cartesian coordinates.

        Returns:
            ArrayLike: Fractional coordinate. A site equivalent to the
                original site, but on the other side of the slab
        """
        spg_analyzer = SpacegroupAnalyzer(self)
        ops = spg_analyzer.get_symmetry_operations(cartesian=cartesian)
        for op in ops:
            slab = self.copy()
            site_other = op.operate(point)
            if f'{site_other[2]:.6f}' == f'{point[2]:.6f}':
                continue
            slab.append('O', point, coords_are_cartesian=cartesian)
            slab.append('O', site_other, coords_are_cartesian=cartesian)
            if SpacegroupAnalyzer(slab).is_laue():
                break
            slab.remove_sites([len(slab) - 1])
            slab.remove_sites([len(slab) - 1])
        return site_other

    def get_orthogonal_c_slab(self) -> Slab:
        """Generate a Slab where the normal (c lattice vector) is
        forced to be orthogonal to the surface a and b lattice vectors.

        **Note that this breaks inherent symmetries in the slab.**

        It should be pointed out that orthogonality is not required to get good
        surface energies, but it can be useful in cases where the slabs are
        subsequently used for postprocessing of some kind, e.g. generating
        grain boundaries or interfaces.
        """
        a, b, c = self.lattice.matrix
        _new_c = np.cross(a, b)
        _new_c /= np.linalg.norm(_new_c)
        new_c = np.dot(c, _new_c) * _new_c
        new_latt = Lattice([a, b, new_c])
        return Slab(lattice=new_latt, species=self.species_and_occu, coords=self.cart_coords, miller_index=self.miller_index, oriented_unit_cell=self.oriented_unit_cell, shift=self.shift, scale_factor=self.scale_factor, coords_are_cartesian=True, energy=self.energy, reorient_lattice=self.reorient_lattice, site_properties=self.site_properties)

    def get_tasker2_slabs(self, tol: float=0.01, same_species_only: bool=True) -> list[Slab]:
        """Get a list of slabs that have been Tasker 2 corrected.

        Args:
            tol (float): Fractional tolerance to determine if atoms are within same plane.
            same_species_only (bool): If True, only those are of the exact same
                species as the atom at the outermost surface are considered for moving.
                Otherwise, all atoms regardless of species within tol are considered for moving.
                Default is True (usually the desired behavior).

        Returns:
            list[Slab]: Tasker 2 corrected slabs.
        """

        def get_equi_index(site: PeriodicSite) -> int:
            """Get the index of the equivalent site for a given site."""
            for idx, equi_sites in enumerate(symm_structure.equivalent_sites):
                if site in equi_sites:
                    return idx
            raise ValueError('Cannot determine equi index!')
        sites = list(self.sites)
        slabs = []
        sorted_csites = sorted(sites, key=lambda site: site.c)
        n_layers_total = int(round(self.lattice.c / self.oriented_unit_cell.lattice.c))
        n_layers_slab = int(round((sorted_csites[-1].c - sorted_csites[0].c) * n_layers_total))
        slab_ratio = n_layers_slab / n_layers_total
        spg_analyzer = SpacegroupAnalyzer(self)
        symm_structure = spg_analyzer.get_symmetrized_structure()
        for surface_site, shift in [(sorted_csites[0], slab_ratio), (sorted_csites[-1], -slab_ratio)]:
            to_move = []
            fixed = []
            for site in sites:
                if abs(site.c - surface_site.c) < tol and (not same_species_only or site.species == surface_site.species):
                    to_move.append(site)
                else:
                    fixed.append(site)
            to_move = sorted(to_move, key=get_equi_index)
            grouped = [list(sites) for k, sites in itertools.groupby(to_move, key=get_equi_index)]
            if len(to_move) == 0 or any((len(g) % 2 != 0 for g in grouped)):
                warnings.warn('Odd number of sites to divide! Try changing the tolerance to ensure even division of sites or create supercells in a or b directions to allow for atoms to be moved!')
                continue
            combinations = []
            for g in grouped:
                combinations.append(list(itertools.combinations(g, int(len(g) / 2))))
            for selection in itertools.product(*combinations):
                species = [site.species for site in fixed]
                frac_coords = [site.frac_coords for site in fixed]
                for struct_matcher in to_move:
                    species.append(struct_matcher.species)
                    for group in selection:
                        if struct_matcher in group:
                            frac_coords.append(struct_matcher.frac_coords)
                            break
                    else:
                        frac_coords.append(struct_matcher.frac_coords + [0, 0, shift])
                sp_fcoord = sorted(zip(species, frac_coords), key=lambda x: x[0])
                species = [x[0] for x in sp_fcoord]
                frac_coords = [x[1] for x in sp_fcoord]
                slab = Slab(self.lattice, species, frac_coords, self.miller_index, self.oriented_unit_cell, self.shift, self.scale_factor, energy=self.energy, reorient_lattice=self.reorient_lattice)
                slabs.append(slab)
        struct_matcher = StructureMatcher()
        return [ss[0] for ss in struct_matcher.group_structures(slabs)]

    def get_sorted_structure(self, key=None, reverse: bool=False) -> Slab:
        """Get a sorted copy of the structure. The parameters have the same
        meaning as in list.sort. By default, sites are sorted by the
        electronegativity of the species. Note that Slab has to override this
        because of the different __init__ args.

        Args:
            key: Specifies a function of one argument that is used to extract
                a comparison key from each list element: key=str.lower. The
                default value is None (compare the elements directly).
            reverse (bool): If set to True, then the list elements are sorted
                as if each comparison were reversed.
        """
        sites = sorted(self, key=key, reverse=reverse)
        struct = Structure.from_sites(sites)
        return Slab(struct.lattice, struct.species_and_occu, struct.frac_coords, self.miller_index, self.oriented_unit_cell, self.shift, self.scale_factor, site_properties=struct.site_properties, reorient_lattice=self.reorient_lattice)

    def add_adsorbate_atom(self, indices: list[int], species: str | Element | Species, distance: float, specie: Species | Element | str | None=None) -> Self:
        """Add adsorbate onto the Slab, along the c lattice vector.

        Args:
            indices (list[int]): Indices of sites on which to put the adsorbate.
                Adsorbate will be placed relative to the center of these sites.
            species (str | Element | Species): The species to add.
            distance (float): between centers of the adsorbed atom and the
                given site in Angstroms, along the c lattice vector.
            specie: Deprecated argument in #3691. Use 'species' instead.

        Returns:
            Slab: self with adsorbed atom.
        """
        if specie is not None:
            warnings.warn("The argument 'specie' is deprecated. Use 'species' instead.", DeprecationWarning)
            species = specie
        center = np.sum([self[idx].coords for idx in indices], axis=0) / len(indices)
        coords = center + self.normal * distance
        self.append(species, coords, coords_are_cartesian=True)
        return self

    def symmetrically_add_atom(self, species: str | Element | Species, point: ArrayLike, specie: str | Element | Species | None=None, coords_are_cartesian: bool=False) -> None:
        """Add a species at a specified site in a slab. Will also add an
        equivalent site on the other side of the slab to maintain symmetry.

        TODO (@DanielYang59): use "site" over "point" as arg name for consistency

        Args:
            species (str | Element | Species): The species to add.
            point (ArrayLike): The coordinate of the target site.
            specie: Deprecated argument name in #3691. Use 'species' instead.
            coords_are_cartesian (bool): If the site is in Cartesian coordinates.
        """
        if specie is not None:
            warnings.warn("The argument 'specie' is deprecated. Use 'species' instead.", DeprecationWarning)
            species = specie
        equi_site = self.get_symmetric_site(point, cartesian=coords_are_cartesian)
        self.append(species, point, coords_are_cartesian=coords_are_cartesian)
        self.append(species, equi_site, coords_are_cartesian=coords_are_cartesian)

    def symmetrically_remove_atoms(self, indices: list[int]) -> None:
        """Remove sites from a list of indices. Will also remove the
        equivalent site on the other side of the slab to maintain symmetry.

        Args:
            indices (list[int]): The indices of the sites to remove.

        TODO(@DanielYang59):
        1. Reuse public method get_symmetric_site to get equi sites?
        2. If not 1, get_equi_sites has multiple nested loops
        """

        def get_equi_sites(slab: Slab, sites: list[int]) -> list[int]:
            """
            Get the indices of the equivalent sites of given sites.

            Parameters:
                slab (Slab): The slab structure.
                sites (list[int]): Original indices of sites.

            Returns:
                list[int]: Indices of the equivalent sites.
            """
            equi_sites = []
            for pt in sites:
                cart_point = slab.lattice.get_cartesian_coords(pt)
                dist = [site.distance_from_point(cart_point) for site in slab]
                site1 = dist.index(min(dist))
                for i, eq_sites in enumerate(slab.equivalent_sites):
                    if slab[site1] in eq_sites:
                        eq_indices = slab.equivalent_indices[i]
                        break
                i1 = eq_indices[eq_sites.index(slab[site1])]
                for i2 in eq_indices:
                    if i2 == i1:
                        continue
                    if slab[i2].frac_coords[2] == slab[i1].frac_coords[2]:
                        continue
                    slab = self.copy()
                    slab.remove_sites([i1, i2])
                    if slab.is_symmetric():
                        equi_sites.append(i2)
                        break
            return equi_sites
        slab_copy = SpacegroupAnalyzer(self.copy()).get_symmetrized_structure()
        sites = [slab_copy[i].frac_coords for i in indices]
        equi_sites = get_equi_sites(slab_copy, sites)
        if len(equi_sites) == len(indices):
            self.remove_sites(indices)
            self.remove_sites(equi_sites)
        else:
            warnings.warn('Equivalent sites could not be found for some indices. Surface unchanged.')