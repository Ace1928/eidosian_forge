from __future__ import annotations
import copy
import itertools
import logging
import math
import warnings
from collections import defaultdict
from collections.abc import Sequence
from fractions import Fraction
from functools import lru_cache
from math import cos, sin
from typing import TYPE_CHECKING, Any, Literal
import numpy as np
import scipy.cluster
import spglib
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import SymmOp
from pymatgen.core.structure import Molecule, PeriodicSite, Structure
from pymatgen.symmetry.structure import SymmetrizedStructure
from pymatgen.util.coord import find_in_coord_list, pbc_diff
from pymatgen.util.due import Doi, due
class SpacegroupAnalyzer:
    """Takes a pymatgen Structure object and a symprec.

    Uses spglib to perform various symmetry finding operations.
    """

    def __init__(self, structure: Structure, symprec: float | None=0.01, angle_tolerance: float=5) -> None:
        """
        Args:
            structure (Structure/IStructure): Structure to find symmetry
            symprec (float): Tolerance for symmetry finding. Defaults to 0.01,
                which is fairly strict and works well for properly refined
                structures with atoms in the proper symmetry coordinates. For
                structures with slight deviations from their proper atomic
                positions (e.g., structures relaxed with electronic structure
                codes), a looser tolerance of 0.1 (the value used in Materials
                Project) is often needed.
            angle_tolerance (float): Angle tolerance for symmetry finding. Defaults to 5 degrees.
        """
        self._symprec = symprec
        self._angle_tol = angle_tolerance
        self._structure = structure
        self._site_props = structure.site_properties
        unique_species: list[Element | Species] = []
        zs = []
        magmoms = []
        for species, group in itertools.groupby(structure, key=lambda s: s.species):
            if species in unique_species:
                ind = unique_species.index(species)
                zs.extend([ind + 1] * len(tuple(group)))
            else:
                unique_species.append(species)
                zs.extend([len(unique_species)] * len(tuple(group)))
        has_explicit_magmoms = 'magmom' in structure.site_properties or any((getattr(specie, 'spin', None) is not None for specie in structure.types_of_species))
        for site in structure:
            if hasattr(site, 'magmom'):
                magmoms.append(site.magmom)
            elif site.is_ordered and getattr(site.specie, 'spin', None) is not None:
                magmoms.append(site.specie.spin)
            elif has_explicit_magmoms:
                magmoms.append(0)
        self._unique_species = unique_species
        self._numbers = zs
        if len(magmoms) > 0:
            self._cell: tuple[Any, ...] = (tuple(map(tuple, structure.lattice.matrix.tolist())), tuple(map(tuple, structure.frac_coords.tolist())), tuple(zs), tuple(map(tuple, magmoms) if isinstance(magmoms[0], Sequence) else magmoms))
        else:
            self._cell = (tuple(map(tuple, structure.lattice.matrix.tolist())), tuple(map(tuple, structure.frac_coords.tolist())), tuple(zs))
        self._space_group_data = _get_symmetry_dataset(self._cell, symprec, angle_tolerance)

    def get_space_group_symbol(self) -> str:
        """Get the spacegroup symbol (e.g., Pnma) for structure.

        Returns:
            str: Spacegroup symbol for structure.
        """
        return self._space_group_data['international']

    def get_space_group_number(self) -> int:
        """Get the international spacegroup number (e.g., 62) for structure.

        Returns:
            int: International spacegroup number for structure.
        """
        return int(self._space_group_data['number'])

    def get_space_group_operations(self) -> SpacegroupOperations:
        """Get the SpacegroupOperations for the Structure.

        Returns:
            SpacegroupOperations object.
        """
        return SpacegroupOperations(self.get_space_group_symbol(), self.get_space_group_number(), self.get_symmetry_operations())

    def get_hall(self) -> str:
        """Returns Hall symbol for structure.

        Returns:
            str: Hall symbol
        """
        return self._space_group_data['hall']

    def get_point_group_symbol(self) -> str:
        """Get the point group associated with the structure.

        Returns:
            Pointgroup: Point group for structure.
        """
        rotations = self._space_group_data['rotations']
        if len(rotations) == 0:
            return '1'
        return spglib.get_pointgroup(rotations)[0].strip()

    def get_crystal_system(self) -> CrystalSystem:
        """Get the crystal system for the structure, e.g., (triclinic, orthorhombic,
        cubic, etc.).

        Raises:
            ValueError: on invalid space group numbers < 1 or > 230.

        Returns:
            str: Crystal system for structure
        """
        n = self._space_group_data['number']
        if not (n == int(n) and 0 < n < 231):
            raise ValueError(f'Received invalid space group {n}')
        if 0 < n < 3:
            return 'triclinic'
        if n < 16:
            return 'monoclinic'
        if n < 75:
            return 'orthorhombic'
        if n < 143:
            return 'tetragonal'
        if n < 168:
            return 'trigonal'
        if n < 195:
            return 'hexagonal'
        return 'cubic'

    def get_lattice_type(self) -> LatticeType:
        """Get the lattice for the structure, e.g., (triclinic, orthorhombic, cubic,
        etc.).This is the same as the crystal system with the exception of the
        hexagonal/rhombohedral lattice.

        Raises:
            ValueError: on invalid space group numbers < 1 or > 230.

        Returns:
            str: Lattice type for structure
        """
        n = self._space_group_data['number']
        system = self.get_crystal_system()
        if n in [146, 148, 155, 160, 161, 166, 167]:
            return 'rhombohedral'
        if system == 'trigonal':
            return 'hexagonal'
        return system

    def get_symmetry_dataset(self):
        """Returns the symmetry dataset as a dict.

        Returns:
            dict: With the following properties:
                number: International space group number
                international: International symbol
                hall: Hall symbol
                transformation_matrix: Transformation matrix from lattice of
                input cell to Bravais lattice L^bravais = L^original * Tmat
                origin shift: Origin shift in the setting of "Bravais lattice"
                rotations, translations: Rotation matrices and translation
                vectors. Space group operations are obtained by
                [(r,t) for r, t in zip(rotations, translations)]
                wyckoffs: Wyckoff letters
        """
        return self._space_group_data

    def _get_symmetry(self):
        """Get the symmetry operations associated with the structure.

        Returns:
            Symmetry operations as a tuple of two equal length sequences.
            (rotations, translations). "rotations" is the numpy integer array
            of the rotation matrices for scaled positions
            "translations" gives the numpy float64 array of the translation
            vectors in scaled positions.
        """
        dct = spglib.get_symmetry(self._cell, symprec=self._symprec, angle_tolerance=self._angle_tol)
        if dct is None:
            symprec = self._symprec
            raise ValueError(f'Symmetry detection failed for structure with formula {self._structure.formula}. Try setting symprec={symprec!r} to a different value.')
        translations = []
        for t in dct['translations']:
            translations.append([float(Fraction(c).limit_denominator(1000)) for c in t])
        translations = np.array(translations)
        translations[np.abs(translations) == 1] = 0
        return (dct['rotations'], translations)

    def get_symmetry_operations(self, cartesian=False):
        """Return symmetry operations as a list of SymmOp objects. By default returns
        fractional coord sym_ops. But Cartesian can be returned too.

        Returns:
            list[SymmOp]: symmetry operations.
        """
        rotation, translation = self._get_symmetry()
        sym_ops = []
        mat = self._structure.lattice.matrix.T
        inv_mat = np.linalg.inv(mat)
        for rot, trans in zip(rotation, translation):
            if cartesian:
                rot = np.dot(mat, np.dot(rot, inv_mat))
                trans = np.dot(trans, self._structure.lattice.matrix)
            op = SymmOp.from_rotation_and_translation(rot, trans)
            sym_ops.append(op)
        return sym_ops

    def get_point_group_operations(self, cartesian=False):
        """Return symmetry operations as a list of SymmOp objects. By default returns
        fractional coord symm ops. But Cartesian can be returned too.

        Args:
            cartesian (bool): Whether to return SymmOps as Cartesian or
                direct coordinate operations.

        Returns:
            list[SymmOp]: Point group symmetry operations.
        """
        rotation, _translation = self._get_symmetry()
        symm_ops = []
        seen = set()
        mat = self._structure.lattice.matrix.T
        inv_mat = self._structure.lattice.inv_matrix.T
        for rot in rotation:
            rot_hash = rot.tobytes()
            if rot_hash in seen:
                continue
            seen.add(rot_hash)
            if cartesian:
                rot = np.dot(mat, np.dot(rot, inv_mat))
            op = SymmOp.from_rotation_and_translation(rot, np.array([0, 0, 0]))
            symm_ops.append(op)
        return symm_ops

    def get_symmetrized_structure(self):
        """Get a symmetrized structure. A symmetrized structure is one where the sites
        have been grouped into symmetrically equivalent groups.

        Returns:
            pymatgen.symmetry.structure.SymmetrizedStructure object.
        """
        sym_dataset = self.get_symmetry_dataset()
        spg_ops = SpacegroupOperations(self.get_space_group_symbol(), self.get_space_group_number(), self.get_symmetry_operations())
        return SymmetrizedStructure(self._structure, spg_ops, sym_dataset['equivalent_atoms'], sym_dataset['wyckoffs'])

    def get_refined_structure(self, keep_site_properties=False):
        """Get the refined structure based on detected symmetry. The refined structure is
        a *conventional* cell setting with atoms moved to the expected symmetry positions.

        Args:
            keep_site_properties (bool): Whether to keep the input site properties (including
                magnetic moments) on the sites that are still present after the refinement. Note:
                This is disabled by default because the magnetic moments are not always directly
                transferable between unit cell definitions. For instance, long-range magnetic
                ordering or antiferromagnetic character may no longer be present (or exist in
                the same way) in the returned structure. If keep_site_properties is True,
                each site retains the same site property as in the original structure without
                further adjustment.

        Returns:
            Refined structure.
        """
        lattice, scaled_positions, numbers = spglib.refine_cell(self._cell, self._symprec, self._angle_tol)
        species = [self._unique_species[i - 1] for i in numbers]
        if keep_site_properties:
            site_properties = {}
            for k, v in self._site_props.items():
                site_properties[k] = [v[i - 1] for i in numbers]
        else:
            site_properties = None
        struct = Structure(lattice, species, scaled_positions, site_properties=site_properties)
        return struct.get_sorted_structure()

    def find_primitive(self, keep_site_properties=False):
        """Find a primitive version of the unit cell.

        Args:
            keep_site_properties (bool): Whether to keep the input site properties (including
                magnetic moments) on the sites that are still present after the refinement. Note:
                This is disabled by default because the magnetic moments are not always directly
                transferable between unit cell definitions. For instance, long-range magnetic
                ordering or antiferromagnetic character may no longer be present (or exist in
                the same way) in the returned structure. If keep_site_properties is True,
                each site retains the same site property as in the original structure without
                further adjustment.

        Returns:
            A primitive cell in the input cell is searched and returned
            as a Structure object. If no primitive cell is found, None is
            returned.
        """
        lattice, scaled_positions, numbers = spglib.find_primitive(self._cell, symprec=self._symprec)
        species = [self._unique_species[i - 1] for i in numbers]
        if keep_site_properties:
            site_properties = {}
            for k, v in self._site_props.items():
                site_properties[k] = [v[i - 1] for i in numbers]
        else:
            site_properties = None
        return Structure(lattice, species, scaled_positions, to_unit_cell=True, site_properties=site_properties).get_reduced_structure()

    def get_ir_reciprocal_mesh(self, mesh=(10, 10, 10), is_shift=(0, 0, 0)):
        """k-point mesh of the Brillouin zone generated taken into account symmetry.The
        method returns the irreducible kpoints of the mesh and their weights.

        Args:
            mesh (3x1 array): The number of kpoint for the mesh needed in
                each direction
            is_shift (3x1 array): Whether to shift the kpoint grid. (1, 1,
            1) means all points are shifted by 0.5, 0.5, 0.5.

        Returns:
            A list of irreducible kpoints and their weights as a list of
            tuples [(ir_kpoint, weight)], with ir_kpoint given
            in fractional coordinates
        """
        shift = np.array([1 if i else 0 for i in is_shift])
        mapping, grid = spglib.get_ir_reciprocal_mesh(np.array(mesh), self._cell, is_shift=shift, symprec=self._symprec)
        results = []
        for i, count in zip(*np.unique(mapping, return_counts=True)):
            results.append(((grid[i] + shift * (0.5, 0.5, 0.5)) / mesh, count))
        return results

    def get_ir_reciprocal_mesh_map(self, mesh=(10, 10, 10), is_shift=(0, 0, 0)):
        """Same as 'get_ir_reciprocal_mesh' but the full grid together with the mapping
        that maps a reducible to an irreducible kpoint is returned.

        Args:
            mesh (3x1 array): The number of kpoint for the mesh needed in
                each direction
            is_shift (3x1 array): Whether to shift the kpoint grid. (1, 1,
            1) means all points are shifted by 0.5, 0.5, 0.5.

        Returns:
            A tuple containing two numpy.ndarray. The first is the mesh in
            fractional coordinates and the second is an array of integers
            that maps all the reducible kpoints from to irreducible ones.
        """
        shift = np.array([1 if i else 0 for i in is_shift])
        mapping, grid = spglib.get_ir_reciprocal_mesh(np.array(mesh), self._cell, is_shift=shift, symprec=self._symprec)
        grid_fractional_coords = (grid + shift * (0.5, 0.5, 0.5)) / mesh
        return (grid_fractional_coords, mapping)

    @cite_conventional_cell_algo
    def get_conventional_to_primitive_transformation_matrix(self, international_monoclinic=True):
        """Gives the transformation matrix to transform a conventional unit cell to a
        primitive cell according to certain standards the standards are defined in
        Setyawan, W., & Curtarolo, S. (2010). High-throughput electronic band structure
        calculations: Challenges and tools. Computational Materials Science, 49(2),
        299-312. doi:10.1016/j.commatsci.2010.05.010.

        Args:
            international_monoclinic (bool): Whether to convert to proper international convention
                such that beta is the non-right angle.

        Returns:
            Transformation matrix to go from conventional to primitive cell
        """
        conv = self.get_conventional_standard_structure(international_monoclinic=international_monoclinic)
        lattice = self.get_lattice_type()
        if 'P' in self.get_space_group_symbol() or lattice == 'hexagonal':
            return np.eye(3)
        if lattice == 'rhombohedral':
            lengths = conv.lattice.lengths
            if abs(lengths[0] - lengths[2]) < 0.0001:
                transf = np.eye
            else:
                transf = np.array([[-1, 1, 1], [2, 1, 1], [-1, -2, 1]], dtype=np.float64) / 3
        elif 'I' in self.get_space_group_symbol():
            transf = np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.float64) / 2
        elif 'F' in self.get_space_group_symbol():
            transf = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.float64) / 2
        elif 'C' in self.get_space_group_symbol() or 'A' in self.get_space_group_symbol():
            if self.get_crystal_system() == 'monoclinic':
                transf = np.array([[1, 1, 0], [-1, 1, 0], [0, 0, 2]], dtype=np.float64) / 2
            else:
                transf = np.array([[1, -1, 0], [1, 1, 0], [0, 0, 2]], dtype=np.float64) / 2
        else:
            transf = np.eye(3)
        return transf

    @cite_conventional_cell_algo
    def get_primitive_standard_structure(self, international_monoclinic=True, keep_site_properties=False):
        """Gives a structure with a primitive cell according to certain standards. The
        standards are defined in Setyawan, W., & Curtarolo, S. (2010). High-throughput
        electronic band structure calculations: Challenges and tools. Computational
        Materials Science, 49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010.

        Args:
            international_monoclinic (bool): Whether to convert to proper international convention
                such that beta is the non-right angle.
            keep_site_properties (bool): Whether to keep the input site properties (including
                magnetic moments) on the sites that are still present after the refinement. Note:
                This is disabled by default because the magnetic moments are not always directly
                transferable between unit cell definitions. For instance, long-range magnetic
                ordering or antiferromagnetic character may no longer be present (or exist in
                the same way) in the returned structure. If keep_site_properties is True,
                each site retains the same site property as in the original structure without
                further adjustment.

        Returns:
            The structure in a primitive standardized cell
        """
        conv = self.get_conventional_standard_structure(international_monoclinic=international_monoclinic, keep_site_properties=keep_site_properties)
        lattice = self.get_lattice_type()
        if 'P' in self.get_space_group_symbol() or lattice == 'hexagonal':
            return conv
        transf = self.get_conventional_to_primitive_transformation_matrix(international_monoclinic=international_monoclinic)
        new_sites = []
        lattice = Lattice(np.dot(transf, conv.lattice.matrix))
        for site in conv:
            new_s = PeriodicSite(site.specie, site.coords, lattice, to_unit_cell=True, coords_are_cartesian=True, properties=site.properties)
            if not any(map(new_s.is_periodic_image, new_sites)):
                new_sites.append(new_s)
        if lattice == 'rhombohedral':
            prim = Structure.from_sites(new_sites)
            lengths = prim.lattice.lengths
            angles = prim.lattice.angles
            a = lengths[0]
            alpha = math.pi * angles[0] / 180
            new_matrix = [[a * cos(alpha / 2), -a * sin(alpha / 2), 0], [a * cos(alpha / 2), a * sin(alpha / 2), 0], [a * cos(alpha) / cos(alpha / 2), 0, a * math.sqrt(1 - cos(alpha) ** 2 / cos(alpha / 2) ** 2)]]
            new_sites = []
            lattice = Lattice(new_matrix)
            for site in prim:
                new_s = PeriodicSite(site.specie, site.frac_coords, lattice, to_unit_cell=True, properties=site.properties)
                if not any(map(new_s.is_periodic_image, new_sites)):
                    new_sites.append(new_s)
            return Structure.from_sites(new_sites)
        return Structure.from_sites(new_sites)

    @cite_conventional_cell_algo
    def get_conventional_standard_structure(self, international_monoclinic=True, keep_site_properties=False):
        """Gives a structure with a conventional cell according to certain standards. The
        standards are defined in Setyawan, W., & Curtarolo, S. (2010). High-throughput
        electronic band structure calculations: Challenges and tools. Computational
        Materials Science, 49(2), 299-312. doi:10.1016/j.commatsci.2010.05.010 They
        basically enforce as much as possible norm(a1)<norm(a2)<norm(a3). NB This is not
        necessarily the same as the standard settings within the International Tables of
        Crystallography, for which get_refined_structure should be used instead.

        Args:
            international_monoclinic (bool): Whether to convert to proper international convention
                such that beta is the non-right angle.
            keep_site_properties (bool): Whether to keep the input site properties (including
                magnetic moments) on the sites that are still present after the refinement. Note:
                This is disabled by default because the magnetic moments are not always directly
                transferable between unit cell definitions. For instance, long-range magnetic
                ordering or antiferromagnetic character may no longer be present (or exist in
                the same way) in the returned structure. If keep_site_properties is True,
                each site retains the same site property as in the original structure without
                further adjustment.

        Returns:
            The structure in a conventional standardized cell
        """
        tol = 1e-05
        struct = self.get_refined_structure(keep_site_properties=keep_site_properties)
        lattice = struct.lattice
        latt_type = self.get_lattice_type()
        sorted_lengths = sorted(lattice.abc)
        sorted_dic = sorted(({'vec': lattice.matrix[i], 'length': lattice.abc[i], 'orig_index': i} for i in range(3)), key=lambda k: k['length'])
        if latt_type in ('orthorhombic', 'cubic'):
            transf = np.zeros(shape=(3, 3))
            if self.get_space_group_symbol().startswith('C'):
                transf[2] = [0, 0, 1]
                a, b = sorted(lattice.abc[:2])
                sorted_dic = sorted(({'vec': lattice.matrix[i], 'length': lattice.abc[i], 'orig_index': i} for i in [0, 1]), key=lambda k: k['length'])
                for idx in range(2):
                    transf[idx][sorted_dic[idx]['orig_index']] = 1
                c = lattice.abc[2]
            elif self.get_space_group_symbol().startswith('A'):
                transf[2] = [1, 0, 0]
                a, b = sorted(lattice.abc[1:])
                sorted_dic = sorted(({'vec': lattice.matrix[i], 'length': lattice.abc[i], 'orig_index': i} for i in [1, 2]), key=lambda k: k['length'])
                for idx in range(2):
                    transf[idx][sorted_dic[idx]['orig_index']] = 1
                c = lattice.abc[0]
            else:
                for idx, dct in enumerate(sorted_dic):
                    transf[idx][dct['orig_index']] = 1
                a, b, c = sorted_lengths
            lattice = Lattice.orthorhombic(a, b, c)
        elif latt_type == 'tetragonal':
            transf = np.zeros(shape=(3, 3))
            a, b, c = sorted_lengths
            for idx, dct in enumerate(sorted_dic):
                transf[idx][dct['orig_index']] = 1
            if abs(b - c) < tol < abs(a - c):
                a, c = (c, a)
                transf = np.dot([[0, 0, 1], [0, 1, 0], [1, 0, 0]], transf)
            lattice = Lattice.tetragonal(a, c)
        elif latt_type in ('hexagonal', 'rhombohedral'):
            a, b, c = lattice.abc
            if np.all(np.abs([a - b, c - b, a - c]) < 0.001):
                struct.make_supercell(((1, -1, 0), (0, 1, -1), (1, 1, 1)))
                a, b, c = sorted(struct.lattice.abc)
            if abs(b - c) < 0.001:
                a, c = (c, a)
            new_matrix = [[a / 2, -a * math.sqrt(3) / 2, 0], [a / 2, a * math.sqrt(3) / 2, 0], [0, 0, c]]
            lattice = Lattice(new_matrix)
            transf = np.eye(3, 3)
        elif latt_type == 'monoclinic':
            if self.get_space_group_operations().int_symbol.startswith('C'):
                transf = np.zeros(shape=(3, 3))
                transf[2] = [0, 0, 1]
                sorted_dic = sorted(({'vec': lattice.matrix[i], 'length': lattice.abc[i], 'orig_index': i} for i in [0, 1]), key=lambda k: k['length'])
                a = sorted_dic[0]['length']
                b = sorted_dic[1]['length']
                c = lattice.abc[2]
                new_matrix = None
                for t in itertools.permutations(list(range(2)), 2):
                    m = lattice.matrix
                    latt2 = Lattice([m[t[0]], m[t[1]], m[2]])
                    lengths = latt2.lengths
                    angles = latt2.angles
                    if angles[0] > 90:
                        a, b, c, alpha, beta, gamma = Lattice([-m[t[0]], -m[t[1]], m[2]]).parameters
                        transf = np.zeros(shape=(3, 3))
                        transf[0][t[0]] = -1
                        transf[1][t[1]] = -1
                        transf[2][2] = 1
                        alpha = math.pi * alpha / 180
                        new_matrix = [[a, 0, 0], [0, b, 0], [0, c * cos(alpha), c * sin(alpha)]]
                        continue
                    if angles[0] < 90:
                        transf = np.zeros(shape=(3, 3))
                        transf[0][t[0]] = 1
                        transf[1][t[1]] = 1
                        transf[2][2] = 1
                        a, b, c = lengths
                        alpha = math.pi * angles[0] / 180
                        new_matrix = [[a, 0, 0], [0, b, 0], [0, c * cos(alpha), c * sin(alpha)]]
                if new_matrix is None:
                    new_matrix = [[a, 0, 0], [0, b, 0], [0, 0, c]]
                    transf = np.zeros(shape=(3, 3))
                    transf[2] = [0, 0, 1]
                    for idx, dct in enumerate(sorted_dic):
                        transf[idx][dct['orig_index']] = 1
            else:
                new_matrix = None
                for t in itertools.permutations(list(range(3)), 3):
                    m = lattice.matrix
                    a, b, c, alpha, beta, gamma = Lattice([m[t[0]], m[t[1]], m[t[2]]]).parameters
                    if alpha > 90 and b < c:
                        a, b, c, alpha, beta, gamma = Lattice([-m[t[0]], -m[t[1]], m[t[2]]]).parameters
                        transf = np.zeros(shape=(3, 3))
                        transf[0][t[0]] = -1
                        transf[1][t[1]] = -1
                        transf[2][t[2]] = 1
                        alpha = math.pi * alpha / 180
                        new_matrix = [[a, 0, 0], [0, b, 0], [0, c * cos(alpha), c * sin(alpha)]]
                        continue
                    if alpha < 90 and b < c:
                        transf = np.zeros(shape=(3, 3))
                        transf[0][t[0]] = 1
                        transf[1][t[1]] = 1
                        transf[2][t[2]] = 1
                        alpha = math.pi * alpha / 180
                        new_matrix = [[a, 0, 0], [0, b, 0], [0, c * cos(alpha), c * sin(alpha)]]
                if new_matrix is None:
                    new_matrix = [[sorted_lengths[0], 0, 0], [0, sorted_lengths[1], 0], [0, 0, sorted_lengths[2]]]
                    transf = np.zeros(shape=(3, 3))
                    for idx, dct in enumerate(sorted_dic):
                        transf[idx][dct['orig_index']] = 1
            if international_monoclinic:
                op = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
                transf = np.dot(op, transf)
                new_matrix = np.dot(op, new_matrix)
                beta = Lattice(new_matrix).beta
                if beta < 90:
                    op = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
                    transf = np.dot(op, transf)
                    new_matrix = np.dot(op, new_matrix)
            lattice = Lattice(new_matrix)
        elif latt_type == 'triclinic':
            struct = struct.get_reduced_structure('LLL')
            lattice = struct.lattice
            a, b, c = lattice.lengths
            alpha, beta, gamma = (math.pi * i / 180 for i in lattice.angles)
            new_matrix = None
            test_matrix = [[a, 0, 0], [b * cos(gamma), b * sin(gamma), 0.0], [c * cos(beta), c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma), c * math.sqrt(sin(gamma) ** 2 - cos(alpha) ** 2 - cos(beta) ** 2 + 2 * cos(alpha) * cos(beta) * cos(gamma)) / sin(gamma)]]

            def is_all_acute_or_obtuse(matrix) -> bool:
                recp_angles = np.array(Lattice(matrix).reciprocal_lattice.angles)
                return all(recp_angles <= 90) or all(recp_angles > 90)
            if is_all_acute_or_obtuse(test_matrix):
                transf = np.eye(3)
                new_matrix = test_matrix
            test_matrix = [[-a, 0, 0], [b * cos(gamma), b * sin(gamma), 0.0], [-c * cos(beta), -c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma), -c * math.sqrt(sin(gamma) ** 2 - cos(alpha) ** 2 - cos(beta) ** 2 + 2 * cos(alpha) * cos(beta) * cos(gamma)) / sin(gamma)]]
            if is_all_acute_or_obtuse(test_matrix):
                transf = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
                new_matrix = test_matrix
            test_matrix = [[-a, 0, 0], [-b * cos(gamma), -b * sin(gamma), 0.0], [c * cos(beta), c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma), c * math.sqrt(sin(gamma) ** 2 - cos(alpha) ** 2 - cos(beta) ** 2 + 2 * cos(alpha) * cos(beta) * cos(gamma)) / sin(gamma)]]
            if is_all_acute_or_obtuse(test_matrix):
                transf = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
                new_matrix = test_matrix
            test_matrix = [[a, 0, 0], [-b * cos(gamma), -b * sin(gamma), 0.0], [-c * cos(beta), -c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma), -c * math.sqrt(sin(gamma) ** 2 - cos(alpha) ** 2 - cos(beta) ** 2 + 2 * cos(alpha) * cos(beta) * cos(gamma)) / sin(gamma)]]
            if is_all_acute_or_obtuse(test_matrix):
                transf = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
                new_matrix = test_matrix
            lattice = Lattice(new_matrix)
        new_coords = np.dot(transf, np.transpose(struct.frac_coords)).T
        new_struct = Structure(lattice, struct.species_and_occu, new_coords, site_properties=struct.site_properties, to_unit_cell=True)
        return new_struct.get_sorted_structure()

    def get_kpoint_weights(self, kpoints, atol=1e-05):
        """Calculate the weights for a list of kpoints.

        Args:
            kpoints (Sequence): Sequence of kpoints. np.arrays is fine. Note
                that the code does not check that the list of kpoints
                provided does not contain duplicates.
            atol (float): Tolerance for fractional coordinates comparisons.

        Returns:
            List of weights, in the SAME order as kpoints.
        """
        kpts = np.array(kpoints)
        shift = []
        mesh = []
        for idx in range(3):
            nonzero = [i for i in kpts[:, idx] if abs(i) > 1e-05]
            if len(nonzero) != len(kpts):
                if not nonzero:
                    mesh.append(1)
                else:
                    m = np.abs(np.round(1 / np.array(nonzero)))
                    mesh.append(int(max(m)))
                shift.append(0)
            else:
                m = np.abs(np.round(0.5 / np.array(nonzero)))
                mesh.append(int(max(m)))
                shift.append(1)
        mapping, grid = spglib.get_ir_reciprocal_mesh(np.array(mesh), self._cell, is_shift=shift, symprec=self._symprec)
        mapping = list(mapping)
        grid = (np.array(grid) + np.array(shift) * (0.5, 0.5, 0.5)) / mesh
        weights = []
        mapped = defaultdict(int)
        for kpt in kpoints:
            for idx, g in enumerate(grid):
                if np.allclose(pbc_diff(kpt, g), (0, 0, 0), atol=atol):
                    mapped[tuple(g)] += 1
                    weights.append(mapping.count(mapping[idx]))
                    break
        if len(mapped) != len(set(mapping)) or not all((v == 1 for v in mapped.values())):
            raise ValueError('Unable to find 1:1 corresponding between input kpoints and irreducible grid!')
        return [w / sum(weights) for w in weights]

    def is_laue(self) -> bool:
        """Check if the point group of the structure has Laue symmetry (centrosymmetry)."""
        laue = ('-1', '2/m', 'mmm', '4/m', '4/mmm', '-3', '-3m', '6/m', '6/mmm', 'm-3', 'm-3m')
        return str(self.get_point_group_symbol()) in laue