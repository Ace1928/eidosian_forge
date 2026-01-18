from __future__ import annotations
import collections
import contextlib
import functools
import inspect
import io
import itertools
import json
import math
import os
import random
import re
import sys
import warnings
from abc import ABC, abstractmethod
from fnmatch import fnmatch
from inspect import isclass
from io import StringIO
from typing import TYPE_CHECKING, Any, Callable, Literal, SupportsIndex, cast, get_args
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from numpy import cross, eye
from numpy.linalg import norm
from ruamel.yaml import YAML
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import expm, polar
from scipy.spatial.distance import squareform
from tabulate import tabulate
from pymatgen.core.bonds import CovalentBond, get_bond_length
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice, get_points_in_spheres
from pymatgen.core.operations import SymmOp
from pymatgen.core.periodic_table import DummySpecies, Element, Species, get_el_sp
from pymatgen.core.sites import PeriodicSite, Site
from pymatgen.core.units import Length, Mass
from pymatgen.electronic_structure.core import Magmom
from pymatgen.symmetry.maggroups import MagneticSpaceGroup
from pymatgen.util.coord import all_distances, get_angle, lattice_points_in_supercell
class IStructure(SiteCollection, MSONable):
    """Basic immutable Structure object with periodicity. Essentially a sequence
    of PeriodicSites having a common lattice. IStructure is made to be
    (somewhat) immutable so that they can function as keys in a dict. To make
    modifications, use the standard Structure object instead. Structure
    extends Sequence and Hashable, which means that in many cases,
    it can be used like any Python sequence. Iterating through a
    structure is equivalent to going through the sites in sequence.
    """

    def __init__(self, lattice: ArrayLike | Lattice, species: Sequence[CompositionLike], coords: Sequence[ArrayLike], charge: float | None=None, validate_proximity: bool=False, to_unit_cell: bool=False, coords_are_cartesian: bool=False, site_properties: dict | None=None, labels: Sequence[str | None] | None=None, properties: dict | None=None) -> None:
        """Create a periodic structure.

        Args:
            lattice (Lattice/3x3 array): The lattice, either as a
                pymatgen.core.Lattice or
                simply as any 2D array. Each row should correspond to a lattice
                vector. E.g., [[10,0,0], [20,10,0], [0,0,30]] specifies a
                lattice with lattice vectors [10,0,0], [20,10,0] and [0,0,30].
            species ([Species]): Sequence of species on each site. Can take in
                flexible input, including:

                i.  A sequence of element / species specified either as string
                    symbols, e.g. ["Li", "Fe2+", "P", ...] or atomic numbers,
                    e.g., (3, 56, ...) or actual Element or Species objects.

                ii. List of dict of elements/species and occupancies, e.g.,
                    [{"Fe" : 0.5, "Mn":0.5}, ...]. This allows the setup of
                    disordered structures.
            coords (Nx3 array): list of fractional/Cartesian coordinates of
                each species.
            charge (int): overall charge of the structure. Defaults to behavior
                in SiteCollection where total charge is the sum of the oxidation
                states.
            validate_proximity (bool): Whether to check if there are sites
                that are less than 0.01 Ang apart. Defaults to False.
            to_unit_cell (bool): Whether to map all sites into the unit cell,
                i.e. fractional coords between 0 and 1. Defaults to False.
            coords_are_cartesian (bool): Set to True if you are providing
                coordinates in Cartesian coordinates. Defaults to False.
            site_properties (dict): Properties associated with the sites as a
                dict of sequences, e.g. {"magmom":[5, 5, 5, 5]}. The sequences
                have to be the same length as the atomic species and
                fractional_coords. Defaults to None for no properties.
            labels (list[str]): Labels associated with the sites as a
                list of strings, e.g. ['Li1', 'Li2']. Must have the same
                length as the species and fractional coords. Defaults to
                None for no labels.
            properties (dict): Properties associated with the whole structure.
                Will be serialized when writing the structure to JSON or YAML but is
                lost when converting to other formats.
        """
        if len(species) != len(coords):
            raise StructureError(f'len(species)={len(species)!r} != len(coords)={len(coords)!r}')
        self._lattice = lattice if isinstance(lattice, Lattice) else Lattice(lattice)
        sites = []
        for idx, specie in enumerate(species):
            prop = None
            if site_properties:
                prop = {key: val[idx] for key, val in site_properties.items() if val is not None}
            label = labels[idx] if labels else None
            site = PeriodicSite(specie, coords[idx], self._lattice, to_unit_cell, coords_are_cartesian=coords_are_cartesian, properties=prop, label=label)
            sites.append(site)
        self._sites: tuple[PeriodicSite, ...] = tuple(sites)
        if validate_proximity and (not self.is_valid()):
            raise StructureError(f'sites are less than {self.DISTANCE_TOLERANCE} Angstrom apart!')
        self._charge = charge
        self._properties = properties or {}

    @classmethod
    def from_sites(cls, sites: list[PeriodicSite], charge: float | None=None, validate_proximity: bool=False, to_unit_cell: bool=False, properties: dict | None=None) -> IStructure:
        """Convenience constructor to make a Structure from a list of sites.

        Args:
            sites: Sequence of PeriodicSites. Sites must have the same
                lattice.
            charge: Charge of structure.
            validate_proximity (bool): Whether to check if there are sites
                that are less than 0.01 Ang apart. Defaults to False.
            to_unit_cell (bool): Whether to translate sites into the unit
                cell.
            properties (dict): Properties associated with the whole structure.
                Will be serialized when writing the structure to JSON or YAML but is
                lost when converting to other formats.

        Raises:
            ValueError: If sites is empty or sites do not have the same lattice.

        Returns:
            IStructure: Note that missing properties are set as None.
        """
        if not sites:
            raise ValueError(f'You need at least 1 site to construct a {cls.__name__}')
        prop_keys: list[str] = []
        props = {}
        labels = [site.label for site in sites]
        lattice = sites[0].lattice
        for idx, site in enumerate(sites):
            if site.lattice != lattice:
                raise ValueError('Sites must belong to the same lattice')
            for key, val in site.properties.items():
                if key not in prop_keys:
                    prop_keys.append(key)
                    props[key] = [None] * len(sites)
                props[key][idx] = val
        for key, val in props.items():
            if any((vv is None for vv in val)):
                warnings.warn(f'Not all sites have property {key}. Missing values are set to None.')
        return cls(lattice, [site.species for site in sites], [site.frac_coords for site in sites], charge=charge, site_properties=props, validate_proximity=validate_proximity, to_unit_cell=to_unit_cell, labels=labels, properties=properties)

    @classmethod
    def from_spacegroup(cls, sg: str | int, lattice: list | np.ndarray | Lattice, species: Sequence[str | Element | Species | DummySpecies | Composition], coords: Sequence[Sequence[float]], site_properties: dict[str, Sequence] | None=None, coords_are_cartesian: bool=False, tol: float=1e-05, labels: Sequence[str | None] | None=None) -> IStructure | Structure:
        """Generate a structure using a spacegroup. Note that only symmetrically
        distinct species and coords should be provided. All equivalent sites
        are generated from the spacegroup operations.

        Args:
            sg (str/int): The spacegroup. If a string, it will be interpreted
                as one of the notations supported by
                pymatgen.symmetry.groups.Spacegroup. E.g., "R-3c" or "Fm-3m".
                If an int, it will be interpreted as an international number.
            lattice (Lattice/3x3 array): The lattice, either as a
                pymatgen.core.Lattice or
                simply as any 2D array. Each row should correspond to a lattice
                vector. E.g., [[10,0,0], [20,10,0], [0,0,30]] specifies a
                lattice with lattice vectors [10,0,0], [20,10,0] and [0,0,30].
                Note that no attempt is made to check that the lattice is
                compatible with the spacegroup specified. This may be
                introduced in a future version.
            species ([Species]): Sequence of species on each site. Can take in
                flexible input, including:

                i.  A sequence of element / species specified either as string
                    symbols, e.g. ["Li", "Fe2+", "P", ...] or atomic numbers,
                    e.g., (3, 56, ...) or actual Element or Species objects.

                ii. List of dict of elements/species and occupancies, e.g.,
                    [{"Fe" : 0.5, "Mn":0.5}, ...]. This allows the setup of
                    disordered structures.
            coords (Nx3 array): list of fractional/cartesian coordinates of
                each species.
            coords_are_cartesian (bool): Set to True if you are providing
                coordinates in Cartesian coordinates. Defaults to False.
            site_properties (dict): Properties associated with the sites as a
                dict of sequences, e.g., {"magmom":[5,5,5,5]}. The sequences
                have to be the same length as the atomic species and
                fractional_coords. Defaults to None for no properties.
            tol (float): A fractional tolerance to deal with numerical
               precision issues in determining if orbits are the same.
            labels (list[str]): Labels associated with the sites as a
                list of strings, e.g. ['Li1', 'Li2']. Must have the same
                length as the species and fractional coords. Defaults to
                None for no labels.
        """
        from pymatgen.symmetry.groups import SpaceGroup
        try:
            num = int(sg)
            spg = SpaceGroup.from_int_number(num)
        except ValueError:
            spg = SpaceGroup(sg)
        lattice = lattice if isinstance(lattice, Lattice) else Lattice(lattice)
        if not spg.is_compatible(lattice):
            raise ValueError(f'Supplied lattice with parameters {lattice.parameters} is incompatible with supplied spacegroup {spg.symbol}!')
        if len(species) != len(coords):
            raise ValueError(f'Supplied species and coords lengths ({len(species)} vs {len(coords)}) are different!')
        frac_coords = lattice.get_fractional_coords(coords) if coords_are_cartesian else np.array(coords, dtype=np.float64)
        props = {} if site_properties is None else site_properties
        all_sp: list[str | Element | Species | DummySpecies | Composition] = []
        all_coords: list[list[float]] = []
        all_site_properties: dict[str, list] = collections.defaultdict(list)
        all_labels: list[str | None] = []
        for idx, (sp, c) in enumerate(zip(species, frac_coords)):
            cc = spg.get_orbit(c, tol=tol)
            all_sp.extend([sp] * len(cc))
            all_coords.extend(cc)
            label = labels[idx] if labels else None
            all_labels.extend([label] * len(cc))
            for k, v in props.items():
                all_site_properties[k].extend([v[idx]] * len(cc))
        return cls(lattice, all_sp, all_coords, site_properties=all_site_properties, labels=all_labels)

    @classmethod
    def from_magnetic_spacegroup(cls, msg: str | MagneticSpaceGroup, lattice: list | np.ndarray | Lattice, species: Sequence[str | Element | Species | DummySpecies | Composition], coords: Sequence[Sequence[float]], site_properties: dict[str, Sequence], coords_are_cartesian: bool=False, tol: float=1e-05, labels: Sequence[str | None] | None=None) -> IStructure | Structure:
        """Generate a structure using a magnetic spacegroup. Note that only
        symmetrically distinct species, coords and magmoms should be provided.]
        All equivalent sites are generated from the spacegroup operations.

        Args:
            msg (str/list/pymatgen.symmetry.maggroups.MagneticSpaceGroup):
                The magnetic spacegroup.
                If a string, it will be interpreted as one of the notations
                supported by MagneticSymmetryGroup, e.g., "R-3'c" or "Fm'-3'm".
                If a list of two ints, it will be interpreted as the number of
                the spacegroup in its Belov, Neronova and Smirnova (BNS) setting.
            lattice (Lattice/3x3 array): The lattice, either as a
                pymatgen.core.Lattice or
                simply as any 2D array. Each row should correspond to a lattice
                vector. E.g., [[10,0,0], [20,10,0], [0,0,30]] specifies a
                lattice with lattice vectors [10,0,0], [20,10,0] and [0,0,30].
                Note that no attempt is made to check that the lattice is
                compatible with the spacegroup specified. This may be
                introduced in a future version.
            species ([Species]): Sequence of species on each site. Can take in
                flexible input, including:
                i.  A sequence of element / species specified either as string
                symbols, e.g. ["Li", "Fe2+", "P", ...] or atomic numbers,
                e.g., (3, 56, ...) or actual Element or Species objects.

                ii. List of dict of elements/species and occupancies, e.g.,
                    [{"Fe" : 0.5, "Mn":0.5}, ...]. This allows the setup of
                    disordered structures.
            coords (Nx3 array): list of fractional/cartesian coordinates of
                each species.
            site_properties (dict): Properties associated with the sites as a
                dict of sequences, e.g., {"magmom":[5,5,5,5]}. The sequences
                have to be the same length as the atomic species and
                fractional_coords. Unlike Structure.from_spacegroup(),
                this argument is mandatory, since magnetic moment information
                has to be included. Note that the *direction* of the supplied
                magnetic moment relative to the crystal is important, even if
                the resulting structure is used for collinear calculations.
            coords_are_cartesian (bool): Set to True if you are providing
                coordinates in Cartesian coordinates. Defaults to False.
            tol (float): A fractional tolerance to deal with numerical
                precision issues in determining if orbits are the same.
            labels (list[str]): Labels associated with the sites as a
                list of strings, e.g. ['Li1', 'Li2']. Must have the same
                length as the species and fractional coords. Defaults to
                None for no labels.

        Returns:
            Structure | IStructure
        """
        if 'magmom' not in site_properties:
            raise ValueError('Magnetic moments have to be defined.')
        magmoms = [Magmom(m) for m in site_properties['magmom']]
        if not isinstance(msg, MagneticSpaceGroup):
            msg = MagneticSpaceGroup(msg)
        lattice = lattice if isinstance(lattice, Lattice) else Lattice(lattice)
        if not msg.is_compatible(lattice):
            raise ValueError(f'Supplied lattice with parameters {lattice.parameters} is incompatible with supplied spacegroup {msg.sg_symbol}!')
        for name, var in (('coords', coords), ('magmoms', magmoms)):
            if len(var) != len(species):
                raise ValueError(f'Length mismatch: len({name})={len(var)} != len(species)={len(species)!r}')
        frac_coords = lattice.get_fractional_coords(coords) if coords_are_cartesian else coords
        all_sp: list[str | Element | Species | DummySpecies | Composition] = []
        all_coords: list[list[float]] = []
        all_magmoms: list[float] = []
        all_site_properties: dict[str, list] = collections.defaultdict(list)
        all_labels: list[str | None] = []
        for idx, (sp, c, m) in enumerate(zip(species, frac_coords, magmoms)):
            cc, mm = msg.get_orbit(c, m, tol=tol)
            all_sp.extend([sp] * len(cc))
            all_coords.extend(cc)
            all_magmoms.extend(mm)
            label = labels[idx] if labels else None
            all_labels.extend([label] * len(cc))
            for k, v in site_properties.items():
                if k != 'magmom':
                    all_site_properties[k].extend([v[idx]] * len(cc))
        all_site_properties['magmom'] = all_magmoms
        return cls(lattice, all_sp, all_coords, site_properties=all_site_properties, labels=all_labels)

    def unset_charge(self) -> None:
        """Reset the charge to None. E.g. to compute it dynamically based on oxidation states."""
        self._charge = None

    @property
    def properties(self) -> dict:
        """Properties associated with the whole Structure. Note that this information is
        only guaranteed to be saved if serializing to native pymatgen output formats (JSON/YAML).
        """
        if (properties := getattr(self, '_properties', None)):
            return properties
        self._properties = {}
        return self._properties

    @properties.setter
    def properties(self, properties: dict) -> None:
        """Sets properties associated with the whole Structure."""
        self._properties = properties

    @property
    def charge(self) -> float:
        """Overall charge of the structure."""
        formal_charge = super().charge
        if self._charge is None:
            return super().charge
        if abs(formal_charge - self._charge) > 1e-08:
            warnings.warn(f'Structure charge ({self._charge}) is set to be not equal to the sum of oxidation states ({formal_charge}). Use Structure.unset_charge() to reset the charge to None.')
        return self._charge

    @property
    def distance_matrix(self) -> np.ndarray:
        """Returns the distance matrix between all sites in the structure. For
        periodic structures, this should return the nearest image distance.
        """
        return self.lattice.get_all_distances(self.frac_coords, self.frac_coords)

    @property
    def lattice(self) -> Lattice:
        """Lattice of the structure."""
        return self._lattice

    @property
    def density(self) -> float:
        """Returns the density in units of g/cm^3."""
        mass = Mass(self.composition.weight, 'amu')
        return mass.to('g') / (self.volume * Length(1, 'ang').to('cm') ** 3)

    @property
    def pbc(self) -> tuple[bool, bool, bool]:
        """Returns the periodicity of the structure."""
        return self._lattice.pbc

    @property
    def is_3d_periodic(self) -> bool:
        """True if the Lattice is periodic in all directions."""
        return self._lattice.is_3d_periodic

    def get_space_group_info(self, symprec: float=0.01, angle_tolerance: float=5.0) -> tuple[str, int]:
        """Convenience method to quickly get the spacegroup of a structure.

        Args:
            symprec (float): Same definition as in SpacegroupAnalyzer.
                Defaults to 1e-2.
            angle_tolerance (float): Same definition as in SpacegroupAnalyzer.
                Defaults to 5 degrees.

        Returns:
            spacegroup_symbol, international_number
        """
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        spg_analyzer = SpacegroupAnalyzer(self, symprec=symprec, angle_tolerance=angle_tolerance)
        return (spg_analyzer.get_space_group_symbol(), spg_analyzer.get_space_group_number())

    def matches(self, other: IStructure | Structure, anonymous: bool=False, **kwargs) -> bool:
        """Check whether this structure is similar to another structure.
        Basically a convenience method to call structure matching.

        Args:
            other (IStructure/Structure): Another structure.
            anonymous (bool): Whether to use anonymous structure matching which allows distinct
                species in one structure to map to another.
            **kwargs: Same **kwargs as in
                pymatgen.analysis.structure_matcher.StructureMatcher.

        Returns:
            bool: True if the structures are similar under some affine transformation.
        """
        from pymatgen.analysis.structure_matcher import StructureMatcher
        matcher = StructureMatcher(**kwargs)
        if anonymous:
            return matcher.fit_anonymous(self, other)
        return matcher.fit(self, other)

    def __eq__(self, other: object) -> bool:
        needed_attrs = ('lattice', 'sites', 'properties')
        if not all((hasattr(other, attr) for attr in needed_attrs)):
            return NotImplemented
        other = cast(Structure, other)
        if other is self:
            return True
        if len(self) != len(other):
            return False
        if self.lattice != other.lattice:
            return False
        if self.properties != other.properties:
            return False
        return all((site in other for site in self))

    def __hash__(self) -> int:
        return hash(self.composition)

    def __mul__(self, scaling_matrix: int | Sequence[int] | Sequence[Sequence[int]]) -> Structure:
        """Makes a supercell. Allowing to have sites outside the unit cell.

        Args:
            scaling_matrix: A scaling matrix for transforming the lattice
                vectors. Has to be all integers. Several options are possible:

                a. A full 3x3 scaling matrix defining the linear combination
                   of the old lattice vectors. E.g., [[2,1,0],[0,3,0],[0,0,
                   1]] generates a new structure with lattice vectors a' =
                   2a + b, b' = 3b, c' = c where a, b, and c are the lattice
                   vectors of the original structure.
                b. A sequence of three scaling factors. E.g., [2, 1, 1]
                   specifies that the supercell should have dimensions 2a x b x
                   c.
                c. A number, which simply scales all lattice vectors by the
                   same factor.

        Returns:
            Supercell structure. Note that a Structure is always returned,
            even if the input structure is a subclass of Structure. This is
            to avoid different arguments signatures from causing problems. If
            you prefer a subclass to return its own type, you need to override
            this method in the subclass.
        """
        scale_matrix = np.array(scaling_matrix, int)
        if scale_matrix.shape != (3, 3):
            scale_matrix = scale_matrix * np.eye(3)
        new_lattice = Lattice(np.dot(scale_matrix, self.lattice.matrix))
        frac_lattice = lattice_points_in_supercell(scale_matrix)
        cart_lattice = new_lattice.get_cartesian_coords(frac_lattice)
        new_sites = []
        for site in self:
            for vec in cart_lattice:
                periodic_site = PeriodicSite(site.species, site.coords + vec, new_lattice, properties=site.properties, coords_are_cartesian=True, to_unit_cell=False, skip_checks=True, label=site.label)
                new_sites.append(periodic_site)
        new_charge = self._charge * np.linalg.det(scale_matrix) if self._charge else None
        return Structure.from_sites(new_sites, charge=new_charge, to_unit_cell=True)

    def __rmul__(self, scaling_matrix):
        """Similar to __mul__ to preserve commutativeness."""
        return self * scaling_matrix

    @property
    def frac_coords(self):
        """Fractional coordinates as a Nx3 numpy array."""
        return np.array([site.frac_coords for site in self])

    @property
    def volume(self) -> float:
        """Returns the volume of the structure in Angstrom^3."""
        return self._lattice.volume

    def get_distance(self, i: int, j: int, jimage=None) -> float:
        """Get distance between site i and j assuming periodic boundary
        conditions. If the index jimage of two sites atom j is not specified it
        selects the jimage nearest to the i atom and returns the distance and
        jimage indices in terms of lattice vector translations if the index
        jimage of atom j is specified it returns the distance between the i
        atom and the specified jimage atom.

        Args:
            i (int): 1st site index
            j (int): 2nd site index
            jimage: Number of lattice translations in each lattice direction.
                Default is None for nearest image.

        Returns:
            distance
        """
        site: PeriodicSite = self[i]
        return site.distance(self[j], jimage)

    def get_sites_in_sphere(self, pt: ArrayLike, r: float, include_index: bool=False, include_image: bool=False) -> list[PeriodicNeighbor]:
        """Find all sites within a sphere from the point, including a site (if any)
        sitting on the point itself. This includes sites in other periodic
        images.

        Algorithm:

        1. place sphere of radius r in crystal and determine minimum supercell
           (parallelepiped) which would contain a sphere of radius r. for this
           we need the projection of a_1 on a unit vector perpendicular
           to a_2 & a_3 (i.e. the unit vector in the direction b_1) to
           determine how many a_1"s it will take to contain the sphere.

           Nxmax = r * length_of_b_1 / (2 Pi)

        2. keep points falling within r.

        Args:
            pt (3x1 array): Cartesian coordinates of center of sphere.
            r (float): Radius of sphere in Angstrom.
            include_index (bool): Whether the non-supercell site index
                is included in the returned data.
            include_image (bool): Whether to include the supercell image
                is included in the returned data.

        Returns:
            PeriodicNeighbor
        """
        neighbors: list[PeriodicNeighbor] = []
        for frac_coord, dist, idx, img in self._lattice.get_points_in_sphere(self.frac_coords, pt, r):
            nn_site = PeriodicNeighbor(self[idx].species, frac_coord, self._lattice, properties=self[idx].properties, nn_distance=dist, image=img, index=idx, label=self[idx].label)
            neighbors.append(nn_site)
        return neighbors

    def get_neighbors(self, site: PeriodicSite, r: float, include_index: bool=False, include_image: bool=False) -> list[PeriodicNeighbor]:
        """Get all neighbors to a site within a sphere of radius r. Excludes the
        site itself.

        Args:
            site (Site): Which is the center of the sphere.
            r (float): Radius of sphere.
            include_index (bool): Deprecated. Now, the non-supercell site index
                is always included in the returned data.
            include_image (bool): Deprecated. Now the supercell image
                is always included in the returned data.

        Returns:
            PeriodicNeighbor
        """
        return self.get_all_neighbors(r, include_index=include_index, include_image=include_image, sites=[site])[0]

    @deprecated(get_neighbors, 'This is retained purely for checking purposes.')
    def get_neighbors_old(self, site, r, include_index=False, include_image=False):
        """Get all neighbors to a site within a sphere of radius r. Excludes the
        site itself.

        Args:
            site (Site): Which is the center of the sphere.
            r (float): Radius of sphere.
            include_index (bool): Whether the non-supercell site index
                is included in the returned data
            include_image (bool): Whether to include the supercell image
                is included in the returned data

        Returns:
            PeriodicNeighbor
        """
        nn = self.get_sites_in_sphere(site.coords, r, include_index=include_index, include_image=include_image)
        return [d for d in nn if site != d[0]]

    def _get_neighbor_list_py(self, r: float, sites: list[PeriodicSite] | None=None, numerical_tol: float=1e-08, exclude_self: bool=True) -> tuple[np.ndarray, ...]:
        """A python version of getting neighbor_list. The returned values are a tuple of
        numpy arrays (center_indices, points_indices, offset_vectors, distances).
        Atom `center_indices[i]` has neighbor atom `points_indices[i]` that is
        translated by `offset_vectors[i]` lattice vectors, and the distance is
        `distances[i]`.

        Args:
            r (float): Radius of sphere
            sites (list of Sites or None): sites for getting all neighbors,
                default is None, which means neighbors will be obtained for all
                sites. This is useful in the situation where you are interested
                only in one subspecies type, and makes it a lot faster.
            numerical_tol (float): This is a numerical tolerance for distances.
                Sites which are < numerical_tol are determined to be coincident
                with the site. Sites which are r + numerical_tol away is deemed
                to be within r from the site. The default of 1e-8 should be
                ok in most instances.
            exclude_self (bool): whether to exclude atom neighboring with itself within
                numerical tolerance distance, default to True

        Returns:
            tuple: (center_indices, points_indices, offset_vectors, distances)
        """
        neighbors = self.get_all_neighbors_py(r=r, include_index=True, include_image=True, sites=sites, numerical_tol=1e-08)
        center_indices = []
        points_indices = []
        offsets = []
        distances = []
        for idx, nns in enumerate(neighbors):
            if len(nns) > 0:
                for nn in nns:
                    if exclude_self and idx == nn.index and (nn.nn_distance <= numerical_tol):
                        continue
                    center_indices.append(idx)
                    points_indices.append(nn.index)
                    offsets.append(nn.image)
                    distances.append(nn.nn_distance)
        return tuple(map(np.array, (center_indices, points_indices, offsets, distances)))

    def get_neighbor_list(self, r: float, sites: Sequence[PeriodicSite] | None=None, numerical_tol: float=1e-08, exclude_self: bool=True) -> tuple[np.ndarray, ...]:
        """Get neighbor lists using numpy array representations without constructing
        Neighbor objects. If the cython extension is installed, this method will
        be orders of magnitude faster than `get_all_neighbors_old` and 2-3x faster
        than `get_all_neighbors`.
        The returned values are a tuple of numpy arrays
        (center_indices, points_indices, offset_vectors, distances).
        Atom `center_indices[i]` has neighbor atom `points_indices[i]` that is
        translated by `offset_vectors[i]` lattice vectors, and the distance is
        `distances[i]`.

        Args:
            r (float): Radius of sphere
            sites (list of Sites or None): sites for getting all neighbors,
                default is None, which means neighbors will be obtained for all
                sites. This is useful in the situation where you are interested
                only in one subspecies type, and makes it a lot faster.
            numerical_tol (float): This is a numerical tolerance for distances.
                Sites which are < numerical_tol are determined to be coincident
                with the site. Sites which are r + numerical_tol away is deemed
                to be within r from the site. The default of 1e-8 should be
                ok in most instances.
            exclude_self (bool): whether to exclude atom neighboring with itself within
                numerical tolerance distance, default to True

        Returns:
            tuple: (center_indices, points_indices, offset_vectors, distances)
        """
        try:
            from pymatgen.optimization.neighbors import find_points_in_spheres
        except ImportError:
            return self._get_neighbor_list_py(r, sites, exclude_self=exclude_self)
        else:
            if sites is None:
                sites = self.sites
            site_coords = np.ascontiguousarray([site.coords for site in sites], dtype=float)
            cart_coords = np.ascontiguousarray(self.cart_coords, dtype=float)
            lattice_matrix = np.ascontiguousarray(self.lattice.matrix, dtype=float)
            pbc = np.ascontiguousarray(self.pbc, dtype=int)
            center_indices, points_indices, images, distances = find_points_in_spheres(cart_coords, site_coords, r=float(r), pbc=pbc, lattice=lattice_matrix, tol=numerical_tol)
            cond = np.array([True] * len(center_indices))
            if exclude_self:
                self_pair = (center_indices == points_indices) & (distances <= numerical_tol)
                cond = ~self_pair
            return (center_indices[cond], points_indices[cond], images[cond], distances[cond])

    def get_symmetric_neighbor_list(self, r: float, sg: str, unique: bool=False, numerical_tol: float=1e-08, exclude_self: bool=True) -> tuple[np.ndarray, ...]:
        """Similar to 'get_neighbor_list' with sites=None, but the neighbors are
        grouped by symmetry. The returned values are a tuple of numpy arrays
        (center_indices, points_indices, offset_vectors, distances, symmetry_indices).
        Atom `center_indices[i]` has neighbor atom `points_indices[i]` that is translated
        by `offset_vectors[i]` lattice vectors, and the distance is `distances[i]`.
        Symmetry_idx groups the bonds that are related by a symmetry of the provided space
        group and symmetry_op is the operation that relates the first bond of the same
        symmetry_idx to the respective atom. The first bond maps onto itself via the
        Identity. The output is sorted w.r.t. to symmetry_indices. If unique is True only
        one of the two bonds connecting two points is given. Out of the two, the bond that
        does not reverse the sites is chosen.

        Args:
            r (float): Radius of sphere
            sg (str/int): The spacegroup the symmetry operations of which will be
                used to classify the neighbors. If a string, it will be interpreted
                as one of the notations supported by
                pymatgen.symmetry.groups.Spacegroup. E.g., "R-3c" or "Fm-3m".
                If an int, it will be interpreted as an international number.
                If None, 'get_space_group_info' will be used to determine the
                space group, default to None.
            unique (bool): Whether a bond is given for both, or only a single
                direction is given. The default is False.
            numerical_tol (float): This is a numerical tolerance for distances.
                Sites which are < numerical_tol are determined to be coincident
                with the site. Sites which are r + numerical_tol away is deemed
                to be within r from the site. The default of 1e-8 should be
                ok in most instances.
            exclude_self (bool): whether to exclude atom neighboring with itself within
                numerical tolerance distance, default to True

        Returns:
            tuple: (center_indices, points_indices, offset_vectors, distances,
                symmetry_indices, symmetry_ops)
        """
        from pymatgen.symmetry.groups import SpaceGroup
        if sg is None:
            ops = SpaceGroup(self.get_space_group_info()[0]).symmetry_ops
        else:
            try:
                sgp = SpaceGroup.from_int_number(int(sg))
            except ValueError:
                sgp = SpaceGroup(sg)
            ops = sgp.symmetry_ops
        lattice = self.lattice
        if not sgp.is_compatible(lattice):
            raise ValueError(f'Supplied lattice with parameters {lattice.parameters} is incompatible with supplied spacegroup {sgp.symbol}!')
        bonds = self.get_neighbor_list(r)
        if unique:
            redundant = []
            for idx, (i, j, R, d) in enumerate(zip(*bonds)):
                if idx in redundant:
                    continue
                for jdx, (i2, j2, R2, d2) in enumerate(zip(*bonds)):
                    bool1 = i == j2
                    bool2 = j == i2
                    bool3 = (-R2 == R).all()
                    bool4 = np.allclose(d, d2, atol=numerical_tol)
                    if bool1 and bool2 and bool3 and bool4:
                        redundant.append(jdx)
            m = ~np.in1d(np.arange(len(bonds[0])), redundant)
            idcs_dist = np.argsort(bonds[3][m])
            bonds = (bonds[0][m][idcs_dist], bonds[1][m][idcs_dist], bonds[2][m][idcs_dist], bonds[3][m][idcs_dist])
        n_bonds = len(bonds[0])
        symmetry_indices = np.empty(n_bonds)
        symmetry_indices[:] = np.nan
        symmetry_ops = np.empty(len(symmetry_indices), dtype=object)
        symmetry_identity = SymmOp.from_rotation_and_translation(np.eye(3), np.zeros(3))
        symmetry_index = 0
        for idx in range(n_bonds):
            if np.isnan(symmetry_indices[idx]):
                symmetry_indices[idx] = symmetry_index
                symmetry_ops[idx] = symmetry_identity
                for jdx in np.arange(n_bonds)[np.isnan(symmetry_indices)]:
                    equal_distance = np.allclose(bonds[3][idx], bonds[3][jdx], atol=numerical_tol)
                    if equal_distance:
                        from_a = self[bonds[0][idx]].frac_coords
                        to_a = self[bonds[1][idx]].frac_coords
                        r_a = bonds[2][idx]
                        from_b = self[bonds[0][jdx]].frac_coords
                        to_b = self[bonds[1][jdx]].frac_coords
                        r_b = bonds[2][jdx]
                        for op in ops:
                            are_related, is_reversed = op.are_symmetrically_related_vectors(from_a, to_a, r_a, from_b, to_b, r_b)
                            if are_related and (not is_reversed):
                                symmetry_indices[jdx] = symmetry_index
                                symmetry_ops[jdx] = op
                            elif are_related and is_reversed:
                                symmetry_indices[jdx] = symmetry_index
                                symmetry_ops[jdx] = op
                                bonds[0][jdx], bonds[1][jdx] = (bonds[1][jdx], bonds[0][jdx])
                                bonds[2][jdx] = -bonds[2][jdx]
                symmetry_index += 1
        idcs_symid = np.argsort(symmetry_indices)
        bonds = (bonds[0][idcs_symid], bonds[1][idcs_symid], bonds[2][idcs_symid], bonds[3][idcs_symid])
        symmetry_indices = symmetry_indices[idcs_symid]
        symmetry_ops = symmetry_ops[idcs_symid]
        idcs_symop = np.arange(n_bonds)
        identity_idcs = np.where(symmetry_ops == symmetry_identity)[0]
        for symmetry_idx in np.unique(symmetry_indices):
            first_idx = np.argmax(symmetry_indices == symmetry_idx)
            for second_idx in identity_idcs:
                if symmetry_indices[second_idx] == symmetry_idx:
                    idcs_symop[first_idx], idcs_symop[second_idx] = (idcs_symop[second_idx], idcs_symop[first_idx])
        return (bonds[0][idcs_symop], bonds[1][idcs_symop], bonds[2][idcs_symop], bonds[3][idcs_symop], symmetry_indices[idcs_symop], symmetry_ops[idcs_symop])

    def get_all_neighbors(self, r: float, include_index: bool=False, include_image: bool=False, sites: Sequence[PeriodicSite] | None=None, numerical_tol: float=1e-08) -> list[list[PeriodicNeighbor]]:
        """Get neighbors for each atom in the unit cell, out to a distance r
        Returns a list of list of neighbors for each site in structure.
        Use this method if you are planning on looping over all sites in the
        crystal. If you only want neighbors for a particular site, use the
        method get_neighbors as it may not have to build such a large supercell
        However if you are looping over all sites in the crystal, this method
        is more efficient since it only performs one pass over a large enough
        supercell to contain all possible atoms out to a distance r.
        The return type is a [(site, dist) ...] since most of the time,
        subsequent processing requires the distance.

        A note about periodic images: Before computing the neighbors, this
        operation translates all atoms to within the unit cell (having
        fractional coordinates within [0,1)). This means that the "image" of a
        site does not correspond to how much it has been translates from its
        current position, but which image of the unit cell it resides.

        Args:
            r (float): Radius of sphere.
            include_index (bool): Deprecated. Now, the non-supercell site index
                is always included in the returned data.
            include_image (bool): Deprecated. Now the supercell image
                is always included in the returned data.
            sites (list of Sites or None): sites for getting all neighbors,
                default is None, which means neighbors will be obtained for all
                sites. This is useful in the situation where you are interested
                only in one subspecies type, and makes it a lot faster.
            numerical_tol (float): This is a numerical tolerance for distances.
                Sites which are < numerical_tol are determined to be coincident
                with the site. Sites which are r + numerical_tol away is deemed
                to be within r from the site. The default of 1e-8 should be
                ok in most instances.

        Returns:
            [[pymatgen.core.structure.PeriodicNeighbor], ..]
        """
        if sites is None:
            sites = self.sites
        center_indices, points_indices, images, distances = self.get_neighbor_list(r=r, sites=sites, numerical_tol=numerical_tol)
        if len(points_indices) < 1:
            return [[]] * len(sites)
        f_coords = self.frac_coords[points_indices] + images
        neighbor_dict: dict[int, list] = collections.defaultdict(list)
        lattice = self.lattice
        atol = Site.position_atol
        all_sites = self.sites
        for cindex, pindex, image, f_coord, d in zip(center_indices, points_indices, images, f_coords, distances):
            psite = all_sites[pindex]
            csite = sites[cindex]
            if d > numerical_tol or psite.species != csite.species or (not np.allclose(psite.coords, csite.coords, atol=atol)) or (psite.properties != csite.properties):
                neighbor_dict[cindex].append(PeriodicNeighbor(species=psite.species, coords=f_coord, lattice=lattice, properties=psite.properties, nn_distance=d, index=pindex, image=tuple(image), label=psite.label))
        neighbors: list[list[PeriodicNeighbor]] = []
        for i in range(len(sites)):
            neighbors.append(neighbor_dict[i])
        return neighbors

    def get_all_neighbors_py(self, r: float, include_index: bool=False, include_image: bool=False, sites: Sequence[PeriodicSite] | None=None, numerical_tol: float=1e-08) -> list[list[PeriodicNeighbor]]:
        """Get neighbors for each atom in the unit cell, out to a distance r
        Returns a list of list of neighbors for each site in structure.
        Use this method if you are planning on looping over all sites in the
        crystal. If you only want neighbors for a particular site, use the
        method get_neighbors as it may not have to build such a large supercell
        However if you are looping over all sites in the crystal, this method
        is more efficient since it only performs one pass over a large enough
        supercell to contain all possible atoms out to a distance r.
        The return type is a [(site, dist) ...] since most of the time,
        subsequent processing requires the distance.

        A note about periodic images: Before computing the neighbors, this
        operation translates all atoms to within the unit cell (having
        fractional coordinates within [0,1)). This means that the "image" of a
        site does not correspond to how much it has been translates from its
        current position, but which image of the unit cell it resides.

        Args:
            r (float): Radius of sphere.
            include_index (bool): Deprecated. Now, the non-supercell site index
                is always included in the returned data.
            include_image (bool): Deprecated. Now the supercell image
                is always included in the returned data.
            sites (list of Sites or None): sites for getting all neighbors,
                default is None, which means neighbors will be obtained for all
                sites. This is useful in the situation where you are interested
                only in one subspecies type, and makes it a lot faster.
            numerical_tol (float): This is a numerical tolerance for distances.
                Sites which are < numerical_tol are determined to be coincident
                with the site. Sites which are r + numerical_tol away is deemed
                to be within r from the site. The default of 1e-8 should be
                ok in most instances.

        Returns:
            list[list[PeriodicNeighbor]]
        """
        if sites is None:
            sites = self.sites
        site_coords = np.array([site.coords for site in sites])
        point_neighbors = get_points_in_spheres(self.cart_coords, site_coords, r=r, pbc=self.pbc, numerical_tol=numerical_tol, lattice=self.lattice)
        neighbors: list[list[PeriodicNeighbor]] = []
        for point_neighbor, site in zip(point_neighbors, sites):
            nns: list[PeriodicNeighbor] = []
            if len(point_neighbor) < 1:
                neighbors.append([])
                continue
            for n in point_neighbor:
                coord, d, index, image = n
                if d > numerical_tol or self[index] != site:
                    neighbor = PeriodicNeighbor(species=self[index].species, coords=coord, lattice=self.lattice, properties=self[index].properties, nn_distance=d, index=index, image=tuple(image))
                    nns.append(neighbor)
            neighbors.append(nns)
        return neighbors

    @deprecated(get_all_neighbors, 'This is retained purely for checking purposes.')
    def get_all_neighbors_old(self, r, include_index=False, include_image=False, include_site=True):
        """Get neighbors for each atom in the unit cell, out to a distance r
        Returns a list of list of neighbors for each site in structure.
        Use this method if you are planning on looping over all sites in the
        crystal. If you only want neighbors for a particular site, use the
        method get_neighbors as it may not have to build such a large supercell
        However if you are looping over all sites in the crystal, this method
        is more efficient since it only performs one pass over a large enough
        supercell to contain all possible atoms out to a distance r.
        The return type is a [(site, dist) ...] since most of the time,
        subsequent processing requires the distance.

        A note about periodic images: Before computing the neighbors, this
        operation translates all atoms to within the unit cell (having
        fractional coordinates within [0,1)). This means that the "image" of a
        site does not correspond to how much it has been translates from its
        current position, but which image of the unit cell it resides.

        Args:
            r (float): Radius of sphere.
            include_index (bool): Whether to include the non-supercell site
                in the returned data
            include_image (bool): Whether to include the supercell image
                in the returned data
            include_site (bool): Whether to include the site in the returned
                data. Defaults to True.

        Returns:
            PeriodicNeighbor
        """
        recp_len = np.array(self.lattice.reciprocal_lattice.abc)
        maxr = np.ceil((r + 0.15) * recp_len / (2 * math.pi))
        nmin = np.floor(np.min(self.frac_coords, axis=0)) - maxr
        nmax = np.ceil(np.max(self.frac_coords, axis=0)) + maxr
        all_ranges = list(itertools.starmap(np.arange, zip(nmin, nmax)))
        lattice = self._lattice
        matrix = lattice.matrix
        neighbors = [[] for _ in range(len(self))]
        all_fcoords = np.mod(self.frac_coords, 1)
        coords_in_cell = np.dot(all_fcoords, matrix)
        site_coords = self.cart_coords
        indices = np.arange(len(self))
        for image in itertools.product(*all_ranges):
            coords = np.dot(image, matrix) + coords_in_cell
            all_dists = all_distances(coords, site_coords)
            all_within_r = np.bitwise_and(all_dists <= r, all_dists > 1e-08)
            for j, d, within_r in zip(indices, all_dists, all_within_r):
                if include_site:
                    nnsite = PeriodicSite(self[j].species, coords[j], lattice, properties=self[j].properties, coords_are_cartesian=True, skip_checks=True)
                for i in indices[within_r]:
                    item = []
                    if include_site:
                        item.append(nnsite)
                    item.append(d[i])
                    if include_index:
                        item.append(j)
                    if include_image:
                        item.append(image)
                    neighbors[i].append(item)
        return neighbors

    def get_neighbors_in_shell(self, origin: ArrayLike, r: float, dr: float, include_index: bool=False, include_image: bool=False) -> list[PeriodicNeighbor]:
        """Returns all sites in a shell centered on origin (coords) between radii
        r-dr and r+dr.

        Args:
            origin (3x1 array): Cartesian coordinates of center of sphere.
            r (float): Inner radius of shell.
            dr (float): Width of shell.
            include_index (bool): Deprecated. Now, the non-supercell site index
                is always included in the returned data.
            include_image (bool): Deprecated. Now the supercell image
                is always included in the returned data.

        Returns:
            [NearestNeighbor] where Nearest Neighbor is a named tuple containing
            (site, distance, index, image).
        """
        outer = self.get_sites_in_sphere(origin, r + dr, include_index=include_index, include_image=include_image)
        inner = r - dr
        return [t for t in outer if t.nn_distance > inner]

    def get_sorted_structure(self, key: Callable | None=None, reverse: bool=False) -> IStructure | Structure:
        """Get a sorted copy of the structure. The parameters have the same
        meaning as in list.sort. By default, sites are sorted by the
        electronegativity of the species.

        Args:
            key: Specifies a function of one argument that is used to extract
                a comparison key from each list element: key=str.lower. The
                default value is None (compare the elements directly).
            reverse (bool): If set to True, then the list elements are sorted
                as if each comparison were reversed.
        """
        sites = sorted(self, key=key, reverse=reverse)
        return type(self).from_sites(sites, charge=self._charge, properties=self.properties)

    def get_reduced_structure(self, reduction_algo: Literal['niggli', 'LLL']='niggli') -> Self:
        """Get a reduced structure.

        Args:
            reduction_algo ("niggli" | "LLL"): The lattice reduction algorithm to use.
                Defaults to "niggli".

        Returns:
            Structure: Niggli- or LLL-reduced structure.
        """
        if reduction_algo == 'niggli':
            reduced_latt = self._lattice.get_niggli_reduced_lattice()
        elif reduction_algo == 'LLL':
            reduced_latt = self._lattice.get_lll_reduced_lattice()
        else:
            raise ValueError(f"Invalid reduction_algo={reduction_algo!r}, must be 'niggli' or 'LLL'.")
        if reduced_latt != self.lattice:
            return type(self)(reduced_latt, self.species_and_occu, self.cart_coords, coords_are_cartesian=True, to_unit_cell=True, site_properties=self.site_properties, labels=self.labels, charge=self._charge)
        return self.copy()

    def copy(self, site_properties: dict[str, Any] | None=None, sanitize: bool=False, properties: dict[str, Any] | None=None) -> Structure:
        """Convenience method to get a copy of the structure, with options to add
        site properties.

        Args:
            site_properties (dict): Properties to add or override. The
                properties are specified in the same way as the constructor,
                i.e., as a dict of the form {property: [values]}. The
                properties should be in the order of the *original* structure
                if you are performing sanitization.
            sanitize (bool): If True, this method will return a sanitized
                structure. Sanitization performs a few things: (i) The sites are
                sorted by electronegativity, (ii) a LLL lattice reduction is
                carried out to obtain a relatively orthogonalized cell,
                (iii) all fractional coords for sites are mapped into the
                unit cell.
            properties (dict): General properties to add or override.

        Returns:
            A copy of the Structure, with optionally new site_properties and
            optionally sanitized.
        """
        new_site_props = self.site_properties
        if site_properties:
            new_site_props.update(site_properties)
        props = self.properties
        if properties:
            props.update(properties)
        if not sanitize:
            return type(self)(self._lattice, self.species_and_occu, self.frac_coords, charge=self._charge, site_properties=new_site_props, labels=self.labels, properties=props)
        reduced_latt = self._lattice.get_lll_reduced_lattice()
        new_sites = []
        for idx, site in enumerate(self):
            frac_coords = reduced_latt.get_fractional_coords(site.coords)
            site_props = {}
            for prop, val in new_site_props.items():
                site_props[prop] = val[idx]
            new_sites.append(PeriodicSite(site.species, frac_coords, reduced_latt, to_unit_cell=True, properties=site_props, label=site.label, skip_checks=True))
        new_sites = sorted(new_sites)
        return type(self).from_sites(new_sites, charge=self._charge, properties=props)

    def interpolate(self, end_structure: IStructure | Structure, nimages: int | Iterable=10, interpolate_lattices: bool=False, pbc: bool=True, autosort_tol: float=0, end_amplitude: float=1) -> list[IStructure | Structure]:
        """Interpolate between this structure and end_structure. Useful for
        construction of NEB inputs. To obtain useful results, the cell setting
        and order of sites must consistent across the start and end structures.

        Args:
            end_structure (Structure): structure to interpolate between this
                structure and end. Must be in the same setting and have the
                same site ordering to yield useful results.
            nimages (int,list): No. of interpolation images or a list of
                interpolation images. Defaults to 10 images.
            interpolate_lattices (bool): Whether to interpolate the lattices.
                Interpolates the lengths and angles (rather than the matrix)
                so orientation may be affected.
            pbc (bool): Whether to use periodic boundary conditions to find
                the shortest path between endpoints.
            autosort_tol (float): A distance tolerance in angstrom in
                which to automatically sort end_structure to match to the
                closest points in this particular structure. This is usually
                what you want in a NEB calculation. 0 implies no sorting.
                Otherwise, a 0.5 value usually works pretty well.
            end_amplitude (float): The fractional amplitude of the endpoint
                of the interpolation, or a cofactor of the distortion vector
                connecting structure to end_structure. Thus, 0 implies no
                distortion, 1 implies full distortion to end_structure
                (default), 0.5 implies distortion to a point halfway
                between structure and end_structure, and -1 implies full
                distortion in the opposite direction to end_structure.

        Returns:
            List of interpolated structures. The starting and ending
            structures included as the first and last structures respectively.
            A total of (nimages + 1) structures are returned.
        """
        if len(self) != len(end_structure):
            raise ValueError('Structures have different lengths!')
        if not (interpolate_lattices or self.lattice == end_structure.lattice):
            raise ValueError('Structures with different lattices!')
        images = nimages if isinstance(nimages, collections.abc.Iterable) else np.arange(nimages + 1) / nimages
        for idx, site in enumerate(self):
            if site.species != end_structure[idx].species:
                raise ValueError(f'Different species!\nStructure 1:\n{self}\nStructure 2\n{end_structure}')
        start_coords = np.array(self.frac_coords)
        end_coords = np.array(end_structure.frac_coords)
        if autosort_tol:
            dist_matrix = self.lattice.get_all_distances(start_coords, end_coords)
            site_mappings: dict[int, list[int]] = collections.defaultdict(list)
            unmapped_start_ind = []
            for idx, row in enumerate(dist_matrix):
                ind = np.where(row < autosort_tol)[0]
                if len(ind) == 1:
                    site_mappings[idx].append(ind[0])
                else:
                    unmapped_start_ind.append(idx)
            if len(unmapped_start_ind) > 1:
                raise ValueError(f'Unable to reliably match structures with autosort_tol = {autosort_tol!r}, unmapped_start_ind = {unmapped_start_ind!r}')
            sorted_end_coords = np.zeros_like(end_coords)
            matched = []
            for idx, j in site_mappings.items():
                if len(j) > 1:
                    raise ValueError(f'Unable to reliably match structures with auto_sort_tol = {autosort_tol}. More than one site match!')
                sorted_end_coords[idx] = end_coords[j[0]]
                matched.append(j[0])
            if len(unmapped_start_ind) == 1:
                idx = unmapped_start_ind[0]
                j = next(iter(set(range(len(start_coords))) - set(matched)))
                sorted_end_coords[idx] = end_coords[j]
            end_coords = sorted_end_coords
        vec = end_amplitude * (end_coords - start_coords)
        if pbc:
            vec[:, self.pbc] -= np.round(vec[:, self.pbc])
        sp = self.species_and_occu
        structs = []
        if interpolate_lattices:
            _u, p = polar(np.dot(end_structure.lattice.matrix.T, np.linalg.inv(self.lattice.matrix.T)))
            lvec = end_amplitude * (p - np.identity(3))
            lstart = self.lattice.matrix.T
        for x in images:
            if interpolate_lattices:
                l_a = np.dot(np.identity(3) + x * lvec, lstart).T
                lattice = Lattice(l_a)
            else:
                lattice = self.lattice
            frac_coords = start_coords + x * vec
            structs.append(type(self)(lattice, sp, frac_coords, site_properties=self.site_properties, labels=self.labels))
        return structs

    def get_miller_index_from_site_indexes(self, site_ids, round_dp=4, verbose=True):
        """Get the Miller index of a plane from a set of sites indexes.

        A minimum of 3 sites are required. If more than 3 sites are given
        the best plane that minimises the distance to all points will be
        calculated.

        Args:
            site_ids (list of int): A list of site indexes to consider. A
                minimum of three site indexes are required. If more than three
                sites are provided, the best plane that minimises the distance
                to all sites will be calculated.
            round_dp (int, optional): The number of decimal places to round the
                miller index to.
            verbose (bool, optional): Whether to print warnings.

        Returns:
            tuple: The Miller index.
        """
        return self.lattice.get_miller_index_from_coords(self.frac_coords[site_ids], coords_are_cartesian=False, round_dp=round_dp, verbose=verbose)

    def get_primitive_structure(self, tolerance: float=0.25, use_site_props: bool=False, constrain_latt: list | dict | None=None):
        """This finds a smaller unit cell than the input. Sometimes it doesn"t
        find the smallest possible one, so this method is recursively called
        until it is unable to find a smaller cell.

        NOTE: if the tolerance is greater than 1/2 the minimum inter-site
        distance in the primitive cell, the algorithm will reject this lattice.

        Args:
            tolerance (float): Tolerance for each coordinate of a
                particular site in Angstroms. For example, [0.1, 0, 0.1] in cartesian
                coordinates will be considered to be on the same coordinates
                as [0, 0, 0] for a tolerance of 0.25. Defaults to 0.25.
            use_site_props (bool): Whether to account for site properties in
                differentiating sites.
            constrain_latt (list/dict): List of lattice parameters we want to
                preserve, e.g. ["alpha", "c"] or dict with the lattice
                parameter names as keys and values we want the parameters to
                be e.g. {"alpha": 90, "c": 2.5}.

        Returns:
            The most primitive structure found.
        """
        if constrain_latt is None:
            constrain_latt = []

        def site_label(site):
            if not use_site_props:
                return site.species_string
            parts = [site.species_string]
            for key in sorted(site.properties):
                parts.append(f'{key}={site.properties[key]}')
            return ', '.join(parts)
        sites = sorted(self._sites, key=site_label)
        grouped_sites = [list(a[1]) for a in itertools.groupby(sites, key=site_label)]
        grouped_fcoords = [np.array([s.frac_coords for s in g]) for g in grouped_sites]
        min_fcoords = min(grouped_fcoords, key=len)
        min_vecs = min_fcoords - min_fcoords[0]
        super_ftol = np.divide(tolerance, self.lattice.abc)
        super_ftol_2 = super_ftol * 2

        def pbc_coord_intersection(fc1, fc2, tol):
            """Returns the fractional coords in fc1 that have coordinates
            within tolerance to some coordinate in fc2.
            """
            dist = fc1[:, None, :] - fc2[None, :, :]
            dist -= np.round(dist)
            return fc1[np.any(np.all(dist < tol, axis=-1), axis=-1)]
        for group in sorted(grouped_fcoords, key=len):
            for frac_coords in group:
                min_vecs = pbc_coord_intersection(min_vecs, group - frac_coords, super_ftol_2)

        def get_hnf(fu):
            """Returns all possible distinct supercell matrices given a
            number of formula units in the supercell. Batches the matrices
            by the values in the diagonal (for less numpy overhead).
            Computational complexity is O(n^3), and difficult to improve.
            Might be able to do something smart with checking combinations of a
            and b first, though unlikely to reduce to O(n^2).
            """

            def factors(n: int):
                for idx in range(1, n + 1):
                    if n % idx == 0:
                        yield idx
            for det in factors(fu):
                if det == 1:
                    continue
                for a in factors(det):
                    for e in factors(det // a):
                        g = det // a // e
                        yield (det, np.array([[[a, b, c], [0, e, f], [0, 0, g]] for b, c, f in itertools.product(range(a), range(a), range(e))]))
        grouped_non_nbrs = []
        for gfcoords in grouped_fcoords:
            fdist = gfcoords[None, :, :] - gfcoords[:, None, :]
            fdist -= np.round(fdist)
            np.abs(fdist, fdist)
            non_nbrs = np.any(fdist > 2 * super_ftol[None, None, :], axis=-1)
            np.fill_diagonal(non_nbrs, val=True)
            grouped_non_nbrs.append(non_nbrs)
        num_fu = functools.reduce(math.gcd, map(len, grouped_sites))
        for size, ms in get_hnf(num_fu):
            inv_ms = np.linalg.inv(ms)
            dist = inv_ms[:, :, None, :] - min_vecs[None, None, :, :]
            dist -= np.round(dist)
            np.abs(dist, dist)
            is_close = np.all(dist < super_ftol, axis=-1)
            any_close = np.any(is_close, axis=-1)
            inds = np.all(any_close, axis=-1)
            for inv_m, m in zip(inv_ms[inds], ms[inds]):
                new_m = np.dot(inv_m, self.lattice.matrix)
                ftol = np.divide(tolerance, np.sqrt(np.sum(new_m ** 2, axis=1)))
                valid = True
                new_coords = []
                new_sp = []
                new_props = collections.defaultdict(list)
                new_labels = []
                for gsites, gfcoords, non_nbrs in zip(grouped_sites, grouped_fcoords, grouped_non_nbrs):
                    all_frac = np.dot(gfcoords, m)
                    fdist = all_frac[None, :, :] - all_frac[:, None, :]
                    fdist = np.abs(fdist - np.round(fdist))
                    close_in_prim = np.all(fdist < ftol[None, None, :], axis=-1)
                    groups = np.logical_and(close_in_prim, non_nbrs)
                    if not np.all(np.sum(groups, axis=0) == size):
                        valid = False
                        break
                    for group in groups:
                        if not np.all(groups[group][:, group]):
                            valid = False
                            break
                    if not valid:
                        break
                    added = np.zeros(len(gsites))
                    new_fcoords = all_frac % 1
                    for i, group in enumerate(groups):
                        if not added[i]:
                            added[group] = True
                            inds = np.where(group)[0]
                            coords = new_fcoords[inds[0]]
                            for n, j in enumerate(inds[1:]):
                                offset = new_fcoords[j] - coords
                                coords += (offset - np.round(offset)) / (n + 2)
                            new_sp.append(gsites[inds[0]].species)
                            for k in gsites[inds[0]].properties:
                                new_props[k].append(gsites[inds[0]].properties[k])
                            new_labels.append(gsites[inds[0]].label)
                            new_coords.append(coords)
                if valid:
                    inv_m = np.linalg.inv(m)
                    new_latt = Lattice(np.dot(inv_m, self.lattice.matrix))
                    struct = Structure(new_latt, new_sp, new_coords, site_properties=new_props, labels=new_labels, coords_are_cartesian=False)
                    p = struct.get_primitive_structure(tolerance=tolerance, use_site_props=use_site_props, constrain_latt=constrain_latt).get_reduced_structure()
                    if not constrain_latt:
                        return p
                    prim_latt, self_latt = (p.lattice, self.lattice)
                    keys = tuple(constrain_latt)
                    is_dict = isinstance(constrain_latt, dict)
                    if np.allclose([getattr(prim_latt, key) for key in keys], [constrain_latt[key] if is_dict else getattr(self_latt, key) for key in keys]):
                        return p
        return self.copy()

    def __repr__(self) -> str:
        outs = ['Structure Summary', repr(self.lattice)]
        if self._charge:
            outs.append(f'Overall Charge: {self._charge:+}')
        for site in self:
            outs.append(repr(site))
        return '\n'.join(outs)

    def __str__(self) -> str:

        def to_str(x) -> str:
            return f'{x:>10.6f}'
        outs = [f'Full Formula ({self.composition.formula})', f'Reduced Formula: {self.composition.reduced_formula}', f'abc   : {' '.join((to_str(i) for i in self.lattice.abc))}', f'angles: {' '.join((to_str(i) for i in self.lattice.angles))}', f'pbc   : {' '.join((str(p).rjust(10) for p in self.lattice.pbc))}']
        if self._charge:
            outs.append(f'Overall Charge: {self._charge:+}')
        outs.append(f'Sites ({len(self)})')
        data = []
        props = self.site_properties
        keys = sorted(props)
        for idx, site in enumerate(self):
            row = [str(idx), site.species_string]
            row.extend([to_str(j) for j in site.frac_coords])
            for key in keys:
                row.append(props[key][idx])
            data.append(row)
        outs.append(tabulate(data, headers=['#', 'SP', 'a', 'b', 'c', *keys]))
        return '\n'.join(outs)

    def get_orderings(self, mode: Literal['enum', 'sqs']='enum', **kwargs) -> list[Structure]:
        """Returns list of orderings for a disordered structure. If structure
        does not contain disorder, the default structure is returned.

        Args:
            mode ("enum" | "sqs"): Either "enum" or "sqs". If enum,
                the enumlib will be used to return all distinct
                orderings. If sqs, mcsqs will be used to return
                an sqs structure.
            kwargs: kwargs passed to either
                pymatgen.command_line..enumlib_caller.EnumlibAdaptor
                or pymatgen.command_line.mcsqs_caller.run_mcsqs.
                For run_mcsqs, a default cluster search of 2 cluster interactions
                with 1NN distance and 3 cluster interactions with 2NN distance
                is set.

        Returns:
            List[Structure]
        """
        if self.is_ordered:
            return [self]
        if mode.startswith('enum'):
            from pymatgen.command_line.enumlib_caller import EnumlibAdaptor
            adaptor = EnumlibAdaptor(self, **kwargs)
            adaptor.run()
            return adaptor.structures
        if mode == 'sqs':
            from pymatgen.command_line.mcsqs_caller import run_mcsqs
            if 'clusters' not in kwargs:
                disordered_sites = [site for site in self if not site.is_ordered]
                subset_structure = Structure.from_sites(disordered_sites)
                dist_matrix = subset_structure.distance_matrix
                dists = sorted(set(dist_matrix.ravel()))
                unique_dists = []
                for idx in range(1, len(dists)):
                    if dists[idx] - dists[idx - 1] > 0.1:
                        unique_dists.append(dists[idx])
                clusters = {idx + 2: dist + 0.01 for idx, dist in enumerate(unique_dists) if idx < 2}
                kwargs['clusters'] = clusters
            return [run_mcsqs(self, **kwargs).bestsqs]
        raise ValueError('Invalid mode!')

    def as_dict(self, verbosity=1, fmt=None, **kwargs) -> dict[str, Any]:
        """Dict representation of Structure.

        Args:
            verbosity (int): Verbosity level. Default of 1 includes both
                direct and Cartesian coordinates for all sites, lattice
                parameters, etc. Useful for reading and for insertion into a
                database. Set to 0 for an extremely lightweight version
                that only includes sufficient information to reconstruct the
                object.
            fmt (str): Specifies a format for the dict. Defaults to None,
                which is the default format used in pymatgen. Other options
                include "abivars".
            **kwargs: Allow passing of other kwargs needed for certain
            formats, e.g., "abivars".

        Returns:
            JSON-serializable dict representation.
        """
        if fmt == 'abivars':
            from pymatgen.io.abinit.abiobjects import structure_to_abivars
            return structure_to_abivars(self, **kwargs)
        latt_dict = self._lattice.as_dict(verbosity=verbosity)
        del latt_dict['@module']
        del latt_dict['@class']
        sites = []
        dct = {'@module': type(self).__module__, '@class': type(self).__name__, 'charge': self.charge, 'lattice': latt_dict, 'properties': self.properties}
        for site in self:
            site_dict = site.as_dict(verbosity=verbosity)
            del site_dict['lattice']
            del site_dict['@module']
            del site_dict['@class']
            sites.append(site_dict)
        dct['sites'] = sites
        return dct

    def as_dataframe(self):
        """Create a Pandas dataframe of the sites. Structure-level attributes are stored in DataFrame.attrs.

        Example:
            Species    a    b             c    x             y             z  magmom
            0    (Si)  0.0  0.0  0.000000e+00  0.0  0.000000e+00  0.000000e+00       5
            1    (Si)  0.0  0.0  1.000000e-7  0.0 -2.217138e-7  3.135509e-7      -5
        """
        import pandas as pd
        data: list[list[str | float]] = []
        site_properties = self.site_properties
        prop_keys = list(site_properties)
        for site in self:
            row = [site.species, *site.frac_coords, *site.coords]
            for key in prop_keys:
                row.append(site.properties.get(key))
            data.append(row)
        df = pd.DataFrame(data, columns=['Species', *'abcxyz', *prop_keys])
        df.attrs['Reduced Formula'] = self.composition.reduced_formula
        df.attrs['Lattice'] = self.lattice
        return df

    @classmethod
    def from_dict(cls, dct: dict[str, Any], fmt: Literal['abivars'] | None=None) -> Self:
        """Reconstitute a Structure object from a dict representation of Structure
        created using as_dict().

        Args:
            dct (dict): Dict representation of structure.
            fmt ('abivars' | None): Use structure_from_abivars() to parse the dict. Defaults to None.

        Returns:
            Structure object
        """
        if fmt == 'abivars':
            from pymatgen.io.abinit.abiobjects import structure_from_abivars
            return structure_from_abivars(cls=cls, **dct)
        lattice = Lattice.from_dict(dct['lattice'])
        sites = [PeriodicSite.from_dict(sd, lattice) for sd in dct['sites']]
        charge = dct.get('charge')
        return cls.from_sites(sites, charge=charge, properties=dct.get('properties'))

    def to(self, filename: str | Path='', fmt: FileFormats='', **kwargs) -> str:
        """Outputs the structure to a file or string.

        Args:
            filename (str): If provided, output will be written to a file. If
                fmt is not specified, the format is determined from the
                filename. Defaults is None, i.e. string output.
            fmt (str): Format to output to. Defaults to JSON unless filename
                is provided. If fmt is specifies, it overrides whatever the
                filename is. Options include "cif", "poscar", "cssr", "json",
                "xsf", "mcsqs", "prismatic", "yaml", "yml", "fleur-inpgen", "pwmat".
                Non-case sensitive.
            **kwargs: Kwargs passthru to relevant methods. E.g., This allows
                the passing of parameters like symprec to the
                CifWriter.__init__ method for generation of symmetric CIFs.

        Returns:
            str: String representation of molecule in given format. If a filename
                is provided, the same string is written to the file.
        """
        filename, fmt = (str(filename), cast(FileFormats, fmt.lower()))
        if fmt == 'cif' or fnmatch(filename.lower(), '*.cif*'):
            from pymatgen.io.cif import CifWriter
            writer = CifWriter(self, **kwargs)
        elif fmt == 'mcif' or fnmatch(filename.lower(), '*.mcif*'):
            from pymatgen.io.cif import CifWriter
            writer = CifWriter(self, write_magmoms=True, **kwargs)
        elif fmt == 'poscar' or fnmatch(filename, '*POSCAR*'):
            from pymatgen.io.vasp import Poscar
            writer = Poscar(self, **kwargs)
        elif fmt == 'cssr' or fnmatch(filename.lower(), '*.cssr*'):
            from pymatgen.io.cssr import Cssr
            writer = Cssr(self)
        elif fmt == 'json' or fnmatch(filename.lower(), '*.json*'):
            json_str = json.dumps(self.as_dict())
            if filename:
                with zopen(filename, mode='wt') as file:
                    file.write(json_str)
            return json_str
        elif fmt == 'xsf' or fnmatch(filename.lower(), '*.xsf*'):
            from pymatgen.io.xcrysden import XSF
            res_str = XSF(self).to_str()
            if filename:
                with zopen(filename, mode='wt', encoding='utf8') as file:
                    file.write(res_str)
            return res_str
        elif fmt == 'mcsqs' or fnmatch(filename, '*rndstr.in*') or fnmatch(filename, '*lat.in*') or fnmatch(filename, '*bestsqs*'):
            from pymatgen.io.atat import Mcsqs
            res_str = Mcsqs(self).to_str()
            if filename:
                with zopen(filename, mode='wt', encoding='ascii') as file:
                    file.write(res_str)
            return res_str
        elif fmt == 'prismatic' or fnmatch(filename, '*prismatic*'):
            from pymatgen.io.prismatic import Prismatic
            return Prismatic(self).to_str()
        elif fmt in ('yaml', 'yml') or fnmatch(filename, '*.yaml*') or fnmatch(filename, '*.yml*'):
            yaml = YAML()
            str_io = StringIO()
            yaml.dump(self.as_dict(), str_io)
            yaml_str = str_io.getvalue()
            if filename:
                with zopen(filename, mode='wt') as file:
                    file.write(yaml_str)
            return yaml_str
        elif fmt == 'fleur-inpgen' or fnmatch(filename, '*.in*'):
            from pymatgen.io.fleur import FleurInput
            writer = FleurInput(self, **kwargs)
        elif fmt == 'res' or fnmatch(filename, '*.res'):
            from pymatgen.io.res import ResIO
            res_str = ResIO.structure_to_str(self)
            if filename:
                with zopen(filename, mode='wt', encoding='utf8') as file:
                    file.write(res_str)
            return res_str
        elif fmt == 'pwmat' or fnmatch(filename.lower(), '*.pwmat') or fnmatch(filename.lower(), '*.config'):
            from pymatgen.io.pwmat import AtomConfig
            writer = AtomConfig(self, **kwargs)
        else:
            if fmt == '':
                raise ValueError(f'Format not specified and could not infer from filename={filename!r}')
            raise ValueError(f'Invalid fmt={fmt!r}, valid options are {get_args(FileFormats)}')
        if filename:
            writer.write_file(filename)
        return str(writer)

    @classmethod
    def from_str(cls, input_string: str, fmt: FileFormats, primitive: bool=False, sort: bool=False, merge_tol: float=0.0, **kwargs) -> Structure | IStructure:
        """Reads a structure from a string.

        Args:
            input_string (str): String to parse.
            fmt (str): A file format specification. One of "cif", "poscar", "cssr",
                "json", "yaml", "yml", "xsf", "mcsqs", "res".
            primitive (bool): Whether to find a primitive cell. Defaults to
                False.
            sort (bool): Whether to sort the sites in accordance to the default
                ordering criteria, i.e., electronegativity.
            merge_tol (float): If this is some positive number, sites that
                are within merge_tol from each other will be merged. Usually
                0.01 should be enough to deal with common numerical issues.
            **kwargs: Passthrough to relevant parser.

        Returns:
            IStructure | Structure
        """
        fmt_low = fmt.lower()
        if fmt_low == 'cif':
            from pymatgen.io.cif import CifParser
            parser = CifParser.from_str(input_string, **kwargs)
            struct = parser.parse_structures(primitive=primitive)[0]
        elif fmt_low == 'poscar':
            from pymatgen.io.vasp import Poscar
            struct = Poscar.from_str(input_string, default_names=False, read_velocities=False, **kwargs).structure
        elif fmt_low == 'cssr':
            from pymatgen.io.cssr import Cssr
            cssr = Cssr.from_str(input_string, **kwargs)
            struct = cssr.structure
        elif fmt_low == 'json':
            dct = json.loads(input_string)
            struct = Structure.from_dict(dct)
        elif fmt_low in ('yaml', 'yml'):
            yaml = YAML()
            dct = yaml.load(input_string)
            struct = Structure.from_dict(dct)
        elif fmt_low == 'xsf':
            from pymatgen.io.xcrysden import XSF
            struct = XSF.from_str(input_string, **kwargs).structure
        elif fmt_low == 'mcsqs':
            from pymatgen.io.atat import Mcsqs
            struct = Mcsqs.structure_from_str(input_string, **kwargs)
        elif fmt == 'fleur-inpgen':
            from pymatgen.io.fleur import FleurInput
            struct = FleurInput.from_string(input_string, inpgen_input=True, **kwargs).structure
        elif fmt == 'fleur':
            from pymatgen.io.fleur import FleurInput
            struct = FleurInput.from_string(input_string, inpgen_input=False).structure
        elif fmt == 'res':
            from pymatgen.io.res import ResIO
            struct = ResIO.structure_from_str(input_string, **kwargs)
        elif fmt == 'pwmat':
            from pymatgen.io.pwmat import AtomConfig
            struct = AtomConfig.from_str(input_string, **kwargs).structure
        else:
            raise ValueError(f'Invalid fmt={fmt!r}, valid options are {get_args(FileFormats)}')
        if sort:
            struct = struct.get_sorted_structure()
        if merge_tol:
            struct.merge_sites(merge_tol)
        return cls.from_sites(struct, properties=struct.properties)

    @classmethod
    def from_file(cls, filename: str | Path, primitive: bool=False, sort: bool=False, merge_tol: float=0.0, **kwargs) -> Structure | IStructure:
        """Reads a structure from a file. For example, anything ending in
        a "cif" is assumed to be a Crystallographic Information Format file.
        Supported formats include CIF, POSCAR/CONTCAR, CHGCAR, LOCPOT,
        vasprun.xml, CSSR, Netcdf and pymatgen's JSON-serialized structures.

        Args:
            filename (str): The filename to read from.
            primitive (bool): Whether to convert to a primitive cell. Defaults to False.
            sort (bool): Whether to sort sites. Default to False.
            merge_tol (float): If this is some positive number, sites that are within merge_tol from each other will be
                merged. Usually 0.01 should be enough to deal with common numerical issues.
            kwargs: Passthrough to relevant reader. E.g. if the file has CIF format, the kwargs will be passed
                through to CifParser.

        Returns:
            Structure.
        """
        filename = str(filename)
        if filename.endswith('.nc'):
            from pymatgen.io.abinit.netcdf import structure_from_ncdata
            struct = structure_from_ncdata(filename, cls=cls)
            if sort:
                struct = struct.get_sorted_structure()
            return struct
        from pymatgen.io.exciting import ExcitingInput
        from pymatgen.io.lmto import LMTOCtrl
        from pymatgen.io.vasp import Chgcar, Vasprun
        fname = os.path.basename(filename)
        with zopen(filename, mode='rt', errors='replace') as file:
            contents = file.read()
        if fnmatch(fname.lower(), '*.cif*') or fnmatch(fname.lower(), '*.mcif*'):
            return cls.from_str(contents, fmt='cif', primitive=primitive, sort=sort, merge_tol=merge_tol, **kwargs)
        if fnmatch(fname, '*POSCAR*') or fnmatch(fname, '*CONTCAR*') or fnmatch(fname, '*.vasp'):
            struct = cls.from_str(contents, fmt='poscar', primitive=primitive, sort=sort, merge_tol=merge_tol, **kwargs)
        elif fnmatch(fname, 'CHGCAR*') or fnmatch(fname, 'LOCPOT*'):
            struct = Chgcar.from_file(filename, **kwargs).structure
        elif fnmatch(fname, 'vasprun*.xml*'):
            struct = Vasprun(filename, **kwargs).final_structure
        elif fnmatch(fname.lower(), '*.cssr*'):
            return cls.from_str(contents, fmt='cssr', primitive=primitive, sort=sort, merge_tol=merge_tol, **kwargs)
        elif fnmatch(fname, '*.json*') or fnmatch(fname, '*.mson*'):
            return cls.from_str(contents, fmt='json', primitive=primitive, sort=sort, merge_tol=merge_tol, **kwargs)
        elif fnmatch(fname, '*.yaml*') or fnmatch(fname, '*.yml*'):
            return cls.from_str(contents, fmt='yaml', primitive=primitive, sort=sort, merge_tol=merge_tol, **kwargs)
        elif fnmatch(fname, '*.xsf'):
            return cls.from_str(contents, fmt='xsf', primitive=primitive, sort=sort, merge_tol=merge_tol, **kwargs)
        elif fnmatch(fname, 'input*.xml'):
            return ExcitingInput.from_file(fname, **kwargs).structure
        elif fnmatch(fname, '*rndstr.in*') or fnmatch(fname, '*lat.in*') or fnmatch(fname, '*bestsqs*'):
            return cls.from_str(contents, fmt='mcsqs', primitive=primitive, sort=sort, merge_tol=merge_tol, **kwargs)
        elif fnmatch(fname, 'CTRL*'):
            return LMTOCtrl.from_file(filename=filename, **kwargs).structure
        elif fnmatch(fname, 'inp*.xml') or fnmatch(fname, '*.in*') or fnmatch(fname, 'inp_*'):
            from pymatgen.io.fleur import FleurInput
            struct = FleurInput.from_file(filename, **kwargs).structure
        elif fnmatch(fname, '*.res'):
            from pymatgen.io.res import ResIO
            struct = ResIO.structure_from_file(filename, **kwargs)
        elif fnmatch(fname.lower(), '*.config*') or fnmatch(fname.lower(), '*.pwmat*'):
            from pymatgen.io.pwmat import AtomConfig
            struct = AtomConfig.from_file(filename, **kwargs).structure
        else:
            raise ValueError(f'Unrecognized extension in filename={filename!r}')
        if sort:
            struct = struct.get_sorted_structure()
        if merge_tol:
            struct.merge_sites(merge_tol)
        struct.__class__ = cls
        return struct
    CellType = Literal['primitive', 'conventional']

    def to_cell(self, cell_type: IStructure.CellType, **kwargs) -> Structure:
        """Returns a cell based on the current structure.

        Args:
            cell_type ("primitive" | "conventional"): Whether to return a primitive or conventional cell.
            kwargs: Any keyword supported by pymatgen.symmetry.analyzer.SpacegroupAnalyzer such as
                symprec=0.01, angle_tolerance=5, international_monoclinic=True and keep_site_properties=False.

        Returns:
            Structure: with the requested cell type.
        """
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        valid_cell_types = get_args(IStructure.CellType)
        if cell_type not in valid_cell_types:
            raise ValueError(f'Invalid cell_type={cell_type!r}, valid values are {valid_cell_types}')
        method_keys = ['international_monoclinic', 'keep_site_properties']
        method_kwargs = {key: kwargs.pop(key) for key in method_keys if key in kwargs}
        sga = SpacegroupAnalyzer(self, **kwargs)
        return getattr(sga, f'get_{cell_type}_standard_structure')(**method_kwargs)

    def to_primitive(self, **kwargs) -> Structure:
        """Returns a primitive cell based on the current structure.

        Args:
            kwargs: Any keyword supported by pymatgen.symmetry.analyzer.SpacegroupAnalyzer such as
                symprec=0.01, angle_tolerance=5, international_monoclinic=True and keep_site_properties=False.

        Returns:
            Structure: with the requested cell type.
        """
        return self.to_cell('primitive', **kwargs)

    def to_conventional(self, **kwargs) -> Structure:
        """Returns a conventional cell based on the current structure.

        Args:
            kwargs: Any keyword supported by pymatgen.symmetry.analyzer.SpacegroupAnalyzer such as
                symprec=0.01, angle_tolerance=5, international_monoclinic=True and keep_site_properties=False.

        Returns:
            Structure: with the requested cell type.
        """
        return self.to_cell('conventional', **kwargs)