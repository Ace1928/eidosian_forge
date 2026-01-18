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
class IMolecule(SiteCollection, MSONable):
    """Basic immutable Molecule object without periodicity. Essentially a
    sequence of sites. IMolecule is made to be immutable so that they can
    function as keys in a dict. For a mutable object, use the Molecule class.

    Molecule extends Sequence and Hashable, which means that in many cases,
    it can be used like any Python sequence. Iterating through a molecule is
    equivalent to going through the sites in sequence.
    """

    def __init__(self, species: Sequence[CompositionLike], coords: Sequence[ArrayLike], charge: float=0.0, spin_multiplicity: int | None=None, validate_proximity: bool=False, site_properties: dict | None=None, labels: Sequence[str | None] | None=None, charge_spin_check: bool=True, properties: dict | None=None) -> None:
        """Create a Molecule.

        Args:
            species: list of atomic species. Possible kinds of input include a
                list of dict of elements/species and occupancies, a List of
                elements/specie specified as actual Element/Species, Strings
                ("Fe", "Fe2+") or atomic numbers (1,56).
            coords (3x1 array): list of Cartesian coordinates of each species.
            charge (float): Charge for the molecule. Defaults to 0.
            spin_multiplicity (int): Spin multiplicity for molecule.
                Defaults to None, which means that the spin multiplicity is
                set to 1 if the molecule has no unpaired electrons and to 2
                if there are unpaired electrons.
            validate_proximity (bool): Whether to check if there are sites
                that are less than 1 Ang apart. Defaults to False.
            site_properties (dict): Properties associated with the sites as
                a dict of sequences, e.g., {"magmom":[5,5,5,5]}. The
                sequences have to be the same length as the atomic species
                and fractional_coords. Defaults to None for no properties.
            labels (list[str]): Labels associated with the sites as a
                list of strings, e.g. ['Li1', 'Li2']. Must have the same
                length as the species and fractional coords. Defaults to
                None for no labels.
            charge_spin_check (bool): Whether to check that the charge and
                spin multiplicity are compatible with each other. Defaults
                to True.
            properties (dict): dictionary containing properties associated
                with the whole molecule.
        """
        if len(species) != len(coords):
            raise StructureError(f'len(species) != len(coords) ({len(species)} != {len(coords)}). List of atomic species must have same length as list of fractional coordinates.')
        self._charge_spin_check = charge_spin_check
        sites: list[Site] = []
        for idx in range(len(species)):
            prop = None
            if site_properties:
                prop = {k: v[idx] for k, v in site_properties.items()}
            label = labels[idx] if labels else None
            sites.append(Site(species[idx], coords[idx], properties=prop, label=label))
        self._sites = tuple(sites)
        if validate_proximity and (not self.is_valid()):
            raise StructureError('Molecule contains sites that are less than 0.01 Angstrom apart!')
        self._charge = charge
        n_electrons = self.nelectrons
        if spin_multiplicity:
            if charge_spin_check and (n_electrons + spin_multiplicity) % 2 != 1:
                raise ValueError(f'Charge of {self._charge} and spin multiplicity of {spin_multiplicity} is not possible for this molecule!')
            self._spin_multiplicity = spin_multiplicity
        else:
            self._spin_multiplicity = 1 if n_electrons % 2 == 0 else 2
        self.properties = properties or {}

    @property
    def charge(self) -> float:
        """Charge of molecule."""
        return self._charge

    @property
    def spin_multiplicity(self) -> float:
        """Spin multiplicity of molecule."""
        return self._spin_multiplicity

    @property
    def nelectrons(self) -> float:
        """Number of electrons in the molecule."""
        n_electrons = 0.0
        for site in self:
            for sp, amt in site.species.items():
                if not isinstance(sp, DummySpecies):
                    n_electrons += sp.Z * amt
        n_electrons -= self.charge
        return n_electrons

    @property
    def center_of_mass(self) -> np.ndarray:
        """Center of mass of molecule."""
        center = np.zeros(3)
        total_weight: float = 0
        for site in self:
            wt = site.species.weight
            center += site.coords * wt
            total_weight += wt
        return center / total_weight

    def copy(self) -> IMolecule | Molecule:
        """Convenience method to get a copy of the molecule.

        Returns:
            IMolecule | Molecule
        """
        return type(self).from_sites(self, properties=self.properties)

    @classmethod
    def from_sites(cls, sites: Sequence[Site], charge: float=0, spin_multiplicity: int | None=None, validate_proximity: bool=False, charge_spin_check: bool=True, properties: dict | None=None) -> IMolecule | Molecule:
        """Convenience constructor to make a Molecule from a list of sites.

        Args:
            sites ([Site]): Sequence of Sites.
            charge (int): Charge of molecule. Defaults to 0.
            spin_multiplicity (int): Spin multicipity. Defaults to None,
                in which it is determined automatically.
            validate_proximity (bool): Whether to check that atoms are too
                close.
            charge_spin_check (bool): Whether to check that the charge and
                spin multiplicity are compatible with each other. Defaults
                to True.
            properties (dict): dictionary containing properties associated
                with the whole molecule.

        Raises:
            ValueError: If sites is empty

        Returns:
            Molecule
        """
        if len(sites) < 1:
            raise ValueError(f'You need at least 1 site to make a {cls.__name__}')
        props = collections.defaultdict(list)
        for site in sites:
            for k, v in site.properties.items():
                props[k].append(v)
        labels = [site.label for site in sites]
        return cls([site.species for site in sites], [site.coords for site in sites], charge=charge, spin_multiplicity=spin_multiplicity, validate_proximity=validate_proximity, site_properties=props, labels=labels, charge_spin_check=charge_spin_check, properties=properties)

    def break_bond(self, ind1: int, ind2: int, tol: float=0.2) -> tuple[IMolecule | Molecule, ...]:
        """Returns two molecules based on breaking the bond between atoms at index
        ind1 and ind2.

        Args:
            ind1 (int): 1st site index
            ind2 (int): 2nd site index
            tol (float): Relative tolerance to test. Basically, the code
                checks if the distance between the sites is less than (1 +
                tol) * typical bond distances. Defaults to 0.2, i.e.,
                20% longer.

        Returns:
            Two Molecule objects representing the two clusters formed from
            breaking the bond.
        """
        clusters = [[self[ind1]], [self[ind2]]]
        sites = [site for idx, site in enumerate(self) if idx not in (ind1, ind2)]

        def belongs_to_cluster(site, cluster):
            return any((CovalentBond.is_bonded(site, test_site, tol=tol) for test_site in cluster))
        while len(sites) > 0:
            unmatched = []
            for site in sites:
                for cluster in clusters:
                    if belongs_to_cluster(site, cluster):
                        cluster.append(site)
                        break
                else:
                    unmatched.append(site)
            if len(unmatched) == len(sites):
                raise ValueError('Not all sites are matched!')
            sites = unmatched
        return tuple((type(self).from_sites(cluster) for cluster in clusters))

    def get_covalent_bonds(self, tol: float=0.2) -> list[CovalentBond]:
        """Determines the covalent bonds in a molecule.

        Args:
            tol (float): The tol to determine bonds in a structure. See
                CovalentBond.is_bonded.

        Returns:
            List of bonds
        """
        bonds = []
        for site1, site2 in itertools.combinations(self._sites, 2):
            if CovalentBond.is_bonded(site1, site2, tol):
                bonds.append(CovalentBond(site1, site2))
        return bonds

    def __eq__(self, other: object) -> bool:
        needed_attrs = ('charge', 'spin_multiplicity', 'sites', 'properties')
        if not all((hasattr(other, attr) for attr in needed_attrs)):
            return NotImplemented
        other = cast(IMolecule, other)
        if len(self) != len(other):
            return False
        if self.charge != other.charge:
            return False
        if self.spin_multiplicity != other.spin_multiplicity:
            return False
        if self.properties != other.properties:
            return False
        return all((site in other for site in self))

    def get_zmatrix(self):
        """Returns a z-matrix representation of the molecule."""
        output = []
        output_var = []
        for idx, site in enumerate(self):
            if idx == 0:
                output.append(f'{site.specie}')
            elif idx == 1:
                nn = self._find_nn_pos_before_site(idx)
                bond_length = self.get_distance(idx, nn[0])
                output.append(f'{self[idx].specie} {nn[0] + 1} B{idx}')
                output_var.append(f'B{idx}={bond_length:.6f}')
            elif idx == 2:
                nn = self._find_nn_pos_before_site(idx)
                bond_length = self.get_distance(idx, nn[0])
                angle = self.get_angle(idx, nn[0], nn[1])
                output.append(f'{self[idx].specie} {nn[0] + 1} B{idx} {nn[1] + 1} A{idx}')
                output_var.extend((f'B{idx}={bond_length:.6f}', f'A{idx}={angle:.6f}'))
            else:
                nn = self._find_nn_pos_before_site(idx)
                bond_length = self.get_distance(idx, nn[0])
                angle = self.get_angle(idx, nn[0], nn[1])
                dih = self.get_dihedral(idx, nn[0], nn[1], nn[2])
                output.append(f'{self[idx].specie} {nn[0] + 1} B{idx} {nn[1] + 1} A{idx} {nn[2] + 1} D{idx}')
                output_var.extend((f'B{idx}={bond_length:.6f}', f'A{idx}={angle:.6f}', f'D{idx}={dih:.6f}'))
        return '\n'.join(output) + '\n\n' + '\n'.join(output_var)

    def _find_nn_pos_before_site(self, site_idx):
        """Returns index of nearest neighbor atoms."""
        all_dist = [(self.get_distance(site_idx, idx), idx) for idx in range(site_idx)]
        all_dist = sorted(all_dist, key=lambda x: x[0])
        return [d[1] for d in all_dist]

    def __hash__(self) -> int:
        return hash(self.composition)

    def __repr__(self) -> str:
        return 'Molecule Summary\n' + '\n'.join(map(repr, self))

    def __str__(self) -> str:
        outs = [f'Full Formula ({self.composition.formula})', 'Reduced Formula: ' + self.composition.reduced_formula, f'Charge = {self._charge}, Spin Mult = {self._spin_multiplicity}', f'Sites ({len(self)})']
        for idx, site in enumerate(self):
            outs.append(f'{idx} {site.species_string} {' '.join([f'{coord:0.6f}'.rjust(12) for coord in site.coords])}')
        return '\n'.join(outs)

    def as_dict(self):
        """JSON-serializable dict representation of Molecule."""
        dct = {'@module': type(self).__module__, '@class': type(self).__name__, 'charge': self.charge, 'spin_multiplicity': self.spin_multiplicity, 'sites': [], 'properties': self.properties}
        for site in self:
            site_dict = site.as_dict()
            del site_dict['@module']
            del site_dict['@class']
            dct['sites'].append(site_dict)
        return dct

    @classmethod
    def from_dict(cls, dct) -> IMolecule | Molecule:
        """Reconstitute a Molecule object from a dict representation created using as_dict().

        Args:
            dct (dict): dict representation of Molecule.

        Returns:
            Molecule
        """
        sites = [Site.from_dict(sd) for sd in dct['sites']]
        charge = dct.get('charge', 0)
        spin_multiplicity = dct.get('spin_multiplicity')
        properties = dct.get('properties')
        return cls.from_sites(sites, charge=charge, spin_multiplicity=spin_multiplicity, properties=properties)

    def get_distance(self, i: int, j: int) -> float:
        """Get distance between site i and j.

        Args:
            i (int): 1st site index
            j (int): 2nd site index

        Returns:
            Distance between the two sites.
        """
        return self[i].distance(self[j])

    def get_sites_in_sphere(self, pt: ArrayLike, r: float) -> list[Neighbor]:
        """Find all sites within a sphere from a point.

        Args:
            pt (3x1 array): Cartesian coordinates of center of sphere
            r (float): Radius of sphere.

        Returns:
            Neighbor
        """
        neighbors = []
        for idx, site in enumerate(self._sites):
            dist = site.distance_from_point(pt)
            if dist <= r:
                neighbors.append(Neighbor(site.species, site.coords, site.properties, dist, idx, label=site.label))
        return neighbors

    def get_neighbors(self, site: Site, r: float) -> list[Neighbor]:
        """Get all neighbors to a site within a sphere of radius r. Excludes the
        site itself.

        Args:
            site (Site): Site at the center of the sphere.
            r (float): Radius of sphere.

        Returns:
            Neighbor
        """
        nns = self.get_sites_in_sphere(site.coords, r)
        return [nn for nn in nns if nn != site]

    def get_neighbors_in_shell(self, origin: ArrayLike, r: float, dr: float) -> list[Neighbor]:
        """Returns all sites in a shell centered on origin (coords) between radii
        r-dr and r+dr.

        Args:
            origin (3x1 array): Cartesian coordinates of center of sphere.
            r (float): Inner radius of shell.
            dr (float): Width of shell.

        Returns:
            Neighbor
        """
        outer = self.get_sites_in_sphere(origin, r + dr)
        inner = r - dr
        return [nn for nn in outer if nn.nn_distance > inner]

    def get_boxed_structure(self, a: float, b: float, c: float, images: ArrayLike=(1, 1, 1), random_rotation: bool=False, min_dist: float=1.0, cls=None, offset: ArrayLike | None=None, no_cross: bool=False, reorder: bool=True) -> IStructure | Structure:
        """Creates a Structure from a Molecule by putting the Molecule in the
        center of a orthorhombic box. Useful for creating Structure for
        calculating molecules using periodic codes.

        Args:
            a (float): a-lattice parameter.
            b (float): b-lattice parameter.
            c (float): c-lattice parameter.
            images: No. of boxed images in each direction. Defaults to
                (1, 1, 1), meaning single molecule with 1 lattice parameter
                in each direction.
            random_rotation (bool): Whether to apply a random rotation to
                each molecule. This jumbles all the molecules so that they
                are not exact images of each other.
            min_dist (float): The minimum distance that atoms should be from
                each other. This is only used if random_rotation is True.
                The randomized rotations are searched such that no two atoms
                are less than min_dist from each other.
            cls: The Structure class to instantiate (defaults to pymatgen
                structure)
            offset: Translation to offset molecule from center of mass coords
            no_cross: Whether to forbid molecule coords from extending beyond
                boundary of box.
            reorder: Whether to reorder the sites to be in electronegativity
                order.

        Returns:
            Structure containing molecule in a box.
        """
        if offset is None:
            offset = np.array([0, 0, 0])
        coords = np.array(self.cart_coords)
        x_range = max(coords[:, 0]) - min(coords[:, 0])
        y_range = max(coords[:, 1]) - min(coords[:, 1])
        z_range = max(coords[:, 2]) - min(coords[:, 2])
        if a <= x_range or b <= y_range or c <= z_range:
            raise ValueError('Box is not big enough to contain Molecule.')
        lattice = Lattice.from_parameters(a * images[0], b * images[1], c * images[2], 90, 90, 90)
        nimages: int = images[0] * images[1] * images[2]
        all_coords: list[ArrayLike] = []
        centered_coords = self.cart_coords - self.center_of_mass + offset
        for i, j, k in itertools.product(list(range(images[0])), list(range(images[1])), list(range(images[2]))):
            box_center = [(i + 0.5) * a, (j + 0.5) * b, (k + 0.5) * c]
            if random_rotation:
                while True:
                    op = SymmOp.from_origin_axis_angle((0, 0, 0), axis=np.random.rand(3), angle=random.uniform(-180, 180))
                    m = op.rotation_matrix
                    new_coords = np.dot(m, centered_coords.T).T + box_center
                    if no_cross:
                        x_max, x_min = (max(new_coords[:, 0]), min(new_coords[:, 0]))
                        y_max, y_min = (max(new_coords[:, 1]), min(new_coords[:, 1]))
                        z_max, z_min = (max(new_coords[:, 2]), min(new_coords[:, 2]))
                        if x_max > a or x_min < 0 or y_max > b or (y_min < 0) or (z_max > c) or (z_min < 0):
                            raise ValueError('Molecule crosses boundary of box')
                    if not all_coords:
                        break
                    distances = lattice.get_all_distances(lattice.get_fractional_coords(new_coords), lattice.get_fractional_coords(all_coords))
                    if np.amin(distances) > min_dist:
                        break
            else:
                new_coords = centered_coords + box_center
                if no_cross:
                    x_max, x_min = (max(new_coords[:, 0]), min(new_coords[:, 0]))
                    y_max, y_min = (max(new_coords[:, 1]), min(new_coords[:, 1]))
                    z_max, z_min = (max(new_coords[:, 2]), min(new_coords[:, 2]))
                    if x_max > a or x_min < 0 or y_max > b or (y_min < 0) or (z_max > c) or (z_min < 0):
                        raise ValueError('Molecule crosses boundary of box')
            all_coords.extend(new_coords)
        sprops = {k: v * nimages for k, v in self.site_properties.items()}
        if cls is None:
            cls = Structure
        if reorder:
            return cls(lattice, self.species * nimages, all_coords, coords_are_cartesian=True, site_properties=sprops, labels=self.labels * nimages).get_sorted_structure()
        return cls(lattice, self.species * nimages, coords, coords_are_cartesian=True, site_properties=sprops, labels=self.labels * nimages)

    def get_centered_molecule(self) -> IMolecule | Molecule:
        """Returns a Molecule centered at the center of mass.

        Returns:
            Molecule centered with center of mass at origin.
        """
        center = self.center_of_mass
        new_coords = np.array(self.cart_coords) - center
        return type(self)(self.species_and_occu, new_coords, charge=self._charge, spin_multiplicity=self._spin_multiplicity, site_properties=self.site_properties, charge_spin_check=self._charge_spin_check, labels=self.labels, properties=self.properties)

    def to(self, filename: str='', fmt: str='') -> str | None:
        """Outputs the molecule to a file or string.

        Args:
            filename (str): If provided, output will be written to a file. If
                fmt is not specified, the format is determined from the
                filename. Defaults is None, i.e. string output.
            fmt (str): Format to output to. Defaults to JSON unless filename
                is provided. If fmt is specifies, it overrides whatever the
                filename is. Options include "xyz", "gjf", "g03", "json". If
                you have OpenBabel installed, any of the formats supported by
                OpenBabel. Non-case sensitive.

        Returns:
            str: String representation of molecule in given format. If a filename
                is provided, the same string is written to the file.
        """
        from pymatgen.io.babel import BabelMolAdaptor
        from pymatgen.io.gaussian import GaussianInput
        from pymatgen.io.xyz import XYZ
        fmt = fmt.lower()
        writer: Any
        if fmt == 'xyz' or fnmatch(filename.lower(), '*.xyz*'):
            writer = XYZ(self)
        elif any((fmt == ext or fnmatch(filename.lower(), f'*.{ext}*') for ext in ('gjf', 'g03', 'g09', 'com', 'inp'))):
            writer = GaussianInput(self)
        elif fmt == 'json' or fnmatch(filename, '*.json*') or fnmatch(filename, '*.mson*'):
            json_str = json.dumps(self.as_dict())
            if filename:
                with zopen(filename, mode='wt', encoding='utf8') as file:
                    file.write(json_str)
            return json_str
        elif fmt in {'yaml', 'yml'} or fnmatch(filename, '*.yaml*') or fnmatch(filename, '*.yml*'):
            yaml = YAML()
            str_io = StringIO()
            yaml.dump(self.as_dict(), str_io)
            yaml_str = str_io.getvalue()
            if filename:
                with zopen(filename, mode='wt', encoding='utf8') as file:
                    file.write(yaml_str)
            return yaml_str
        else:
            match = re.search('\\.(pdb|mol|mdl|sdf|sd|ml2|sy2|mol2|cml|mrv)', filename.lower())
            if not fmt and match:
                fmt = match.group(1)
            writer = BabelMolAdaptor(self)
            return writer.write_file(filename, file_format=fmt)
        if filename:
            writer.write_file(filename)
        return str(writer)

    @classmethod
    def from_str(cls, input_string: str, fmt: Literal['xyz', 'gjf', 'g03', 'g09', 'com', 'inp', 'json', 'yaml']) -> IMolecule | Molecule:
        """Reads the molecule from a string.

        Args:
            input_string (str): String to parse.
            fmt (str): Format to output to. Defaults to JSON unless filename
                is provided. If fmt is specifies, it overrides whatever the
                filename is. Options include "xyz", "gjf", "g03", "json". If
                you have OpenBabel installed, any of the formats supported by
                OpenBabel. Non-case sensitive.

        Returns:
            IMolecule or Molecule.
        """
        from pymatgen.io.gaussian import GaussianInput
        from pymatgen.io.xyz import XYZ
        fmt = fmt.lower()
        if fmt == 'xyz':
            mol = XYZ.from_str(input_string).molecule
        elif fmt in ['gjf', 'g03', 'g09', 'com', 'inp']:
            mol = GaussianInput.from_str(input_string).molecule
        elif fmt == 'json':
            dct = json.loads(input_string)
            return cls.from_dict(dct)
        elif fmt in ('yaml', 'yml'):
            yaml = YAML()
            dct = yaml.load(input_string)
            return cls.from_dict(dct)
        else:
            from pymatgen.io.babel import BabelMolAdaptor
            mol = BabelMolAdaptor.from_str(input_string, file_format=fmt).pymatgen_mol
        return cls.from_sites(mol, properties=mol.properties)

    @classmethod
    def from_file(cls, filename: str | Path) -> Self | None:
        """Reads a molecule from a file. Supported formats include xyz,
        gaussian input (gjf|g03|g09|com|inp), Gaussian output (.out|and
        pymatgen's JSON-serialized molecules. Using openbabel,
        many more extensions are supported but requires openbabel to be
        installed.

        Args:
            filename (str | Path): The filename to read from.

        Returns:
            Molecule
        """
        filename = str(filename)
        from pymatgen.io.gaussian import GaussianOutput
        with zopen(filename) as file:
            contents = file.read()
        fname = filename.lower()
        if fnmatch(fname, '*.xyz*'):
            return cls.from_str(contents, fmt='xyz')
        if any((fnmatch(fname.lower(), f'*.{r}*') for r in ['gjf', 'g03', 'g09', 'com', 'inp'])):
            return cls.from_str(contents, fmt='g09')
        if any((fnmatch(fname.lower(), f'*.{r}*') for r in ['out', 'lis', 'log'])):
            return GaussianOutput(filename).final_structure
        if fnmatch(fname, '*.json*') or fnmatch(fname, '*.mson*'):
            return cls.from_str(contents, fmt='json')
        if fnmatch(fname, '*.yaml*') or fnmatch(filename, '*.yml*'):
            return cls.from_str(contents, fmt='yaml')
        from pymatgen.io.babel import BabelMolAdaptor
        if (match := re.search('\\.(pdb|mol|mdl|sdf|sd|ml2|sy2|mol2|cml|mrv)', filename.lower())):
            new = BabelMolAdaptor.from_file(filename, match.group(1)).pymatgen_mol
            new.__class__ = cls
            return new
        raise ValueError('Cannot determine file type.')