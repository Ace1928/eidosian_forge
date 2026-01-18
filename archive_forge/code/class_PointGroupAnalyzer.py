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
class PointGroupAnalyzer:
    """A class to analyze the point group of a molecule.

    The general outline of the algorithm is as follows:

    1. Center the molecule around its center of mass.
    2. Compute the inertia tensor and the eigenvalues and eigenvectors.
    3. Handle the symmetry detection based on eigenvalues.

        a. Linear molecules have one zero eigenvalue. Possible symmetry
           operations are C*v or D*v
        b. Asymmetric top molecules have all different eigenvalues. The
           maximum rotational symmetry in such molecules is 2
        c. Symmetric top molecules have 1 unique eigenvalue, which gives a
           unique rotation axis. All axial point groups are possible
           except the cubic groups (T & O) and I.
        d. Spherical top molecules have all three eigenvalues equal. They
           have the rare T, O or I point groups.

    Attribute:
        sch_symbol (str): Schoenflies symbol of the detected point group.
    """
    inversion_op = SymmOp.inversion()

    def __init__(self, mol, tolerance=0.3, eigen_tolerance=0.01, matrix_tolerance=0.1):
        """The default settings are usually sufficient.

        Args:
            mol (Molecule): Molecule to determine point group for.
            tolerance (float): Distance tolerance to consider sites as
                symmetrically equivalent. Defaults to 0.3 Angstrom.
            eigen_tolerance (float): Tolerance to compare eigen values of
                the inertia tensor. Defaults to 0.01.
            matrix_tolerance (float): Tolerance used to generate the full set of
                symmetry operations of the point group.
        """
        self.mol = mol
        self.centered_mol = mol.get_centered_molecule()
        self.tol = tolerance
        self.eig_tol = eigen_tolerance
        self.mat_tol = matrix_tolerance
        self._analyze()
        if self.sch_symbol in ['C1v', 'C1h']:
            self.sch_symbol = 'Cs'

    def _analyze(self):
        if len(self.centered_mol) == 1:
            self.sch_symbol = 'Kh'
        else:
            inertia_tensor = np.zeros((3, 3))
            total_inertia = 0
            for site in self.centered_mol:
                c = site.coords
                wt = site.species.weight
                for i in range(3):
                    inertia_tensor[i, i] += wt * (c[(i + 1) % 3] ** 2 + c[(i + 2) % 3] ** 2)
                for i, j in [(0, 1), (1, 2), (0, 2)]:
                    inertia_tensor[i, j] += -wt * c[i] * c[j]
                    inertia_tensor[j, i] += -wt * c[j] * c[i]
                total_inertia += wt * np.dot(c, c)
            inertia_tensor /= total_inertia
            eigvals, eigvecs = np.linalg.eig(inertia_tensor)
            self.principal_axes = eigvecs.T
            self.eigvals = eigvals
            v1, v2, v3 = eigvals
            eig_zero = abs(v1 * v2 * v3) < self.eig_tol
            eig_all_same = abs(v1 - v2) < self.eig_tol and abs(v1 - v3) < self.eig_tol
            eig_all_diff = abs(v1 - v2) > self.eig_tol and abs(v1 - v3) > self.eig_tol and (abs(v2 - v3) > self.eig_tol)
            self.rot_sym = []
            self.symmops = [SymmOp(np.eye(4))]
            if eig_zero:
                logger.debug('Linear molecule detected')
                self._proc_linear()
            elif eig_all_same:
                logger.debug('Spherical top molecule detected')
                self._proc_sph_top()
            elif eig_all_diff:
                logger.debug('Asymmetric top molecule detected')
                self._proc_asym_top()
            else:
                logger.debug('Symmetric top molecule detected')
                self._proc_sym_top()

    def _proc_linear(self):
        if self.is_valid_op(PointGroupAnalyzer.inversion_op):
            self.sch_symbol = 'D*h'
            self.symmops.append(PointGroupAnalyzer.inversion_op)
        else:
            self.sch_symbol = 'C*v'

    def _proc_asym_top(self):
        """Handles asymmetric top molecules, which cannot contain rotational symmetry
        larger than 2.
        """
        self._check_R2_axes_asym()
        if len(self.rot_sym) == 0:
            logger.debug('No rotation symmetries detected.')
            self._proc_no_rot_sym()
        elif len(self.rot_sym) == 3:
            logger.debug('Dihedral group detected.')
            self._proc_dihedral()
        else:
            logger.debug('Cyclic group detected.')
            self._proc_cyclic()

    def _proc_sym_top(self):
        """Handles symmetric top molecules which has one unique eigenvalue whose
        corresponding principal axis is a unique rotational axis.

        More complex handling required to look for R2 axes perpendicular to this unique
        axis.
        """
        if abs(self.eigvals[0] - self.eigvals[1]) < self.eig_tol:
            ind = 2
        elif abs(self.eigvals[1] - self.eigvals[2]) < self.eig_tol:
            ind = 0
        else:
            ind = 1
        logger.debug(f'Eigenvalues = {self.eigvals}.')
        unique_axis = self.principal_axes[ind]
        self._check_rot_sym(unique_axis)
        logger.debug(f'Rotation symmetries = {self.rot_sym}')
        if len(self.rot_sym) > 0:
            self._check_perpendicular_r2_axis(unique_axis)
        if len(self.rot_sym) >= 2:
            self._proc_dihedral()
        elif len(self.rot_sym) == 1:
            self._proc_cyclic()
        else:
            self._proc_no_rot_sym()

    def _proc_no_rot_sym(self):
        """Handles molecules with no rotational symmetry.

        Only possible point groups are C1, Cs and Ci.
        """
        self.sch_symbol = 'C1'
        if self.is_valid_op(PointGroupAnalyzer.inversion_op):
            self.sch_symbol = 'Ci'
            self.symmops.append(PointGroupAnalyzer.inversion_op)
        else:
            for v in self.principal_axes:
                mirror_type = self._find_mirror(v)
                if mirror_type != '':
                    self.sch_symbol = 'Cs'
                    break

    def _proc_cyclic(self):
        """Handles cyclic group molecules."""
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        self.sch_symbol = f'C{rot}'
        mirror_type = self._find_mirror(main_axis)
        if mirror_type == 'h':
            self.sch_symbol += 'h'
        elif mirror_type == 'v':
            self.sch_symbol += 'v'
        elif mirror_type == '' and self.is_valid_op(SymmOp.rotoreflection(main_axis, angle=180 / rot)):
            self.sch_symbol = f'S{2 * rot}'

    def _proc_dihedral(self):
        """Handles dihedral group molecules, i.e those with intersecting R2 axes and a
        main axis.
        """
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        self.sch_symbol = f'D{rot}'
        mirror_type = self._find_mirror(main_axis)
        if mirror_type == 'h':
            self.sch_symbol += 'h'
        elif mirror_type != '':
            self.sch_symbol += 'd'

    def _check_R2_axes_asym(self):
        """Test for 2-fold rotation along the principal axes.

        Used to handle asymmetric top molecules.
        """
        for v in self.principal_axes:
            op = SymmOp.from_axis_angle_and_translation(v, 180)
            if self.is_valid_op(op):
                self.symmops.append(op)
                self.rot_sym.append((v, 2))

    def _find_mirror(self, axis):
        """Looks for mirror symmetry of specified type about axis.

        Possible types are "h" or "vd". Horizontal (h) mirrors are perpendicular to the
        axis while vertical (v) or diagonal (d) mirrors are parallel. v mirrors has atoms
        lying on the mirror plane while d mirrors do not.
        """
        mirror_type = ''
        if self.is_valid_op(SymmOp.reflection(axis)):
            self.symmops.append(SymmOp.reflection(axis))
            mirror_type = 'h'
        else:
            for s1, s2 in itertools.combinations(self.centered_mol, 2):
                if s1.species == s2.species:
                    normal = s1.coords - s2.coords
                    if np.dot(normal, axis) < self.tol:
                        op = SymmOp.reflection(normal)
                        if self.is_valid_op(op):
                            self.symmops.append(op)
                            if len(self.rot_sym) > 1:
                                mirror_type = 'd'
                                for v, _ in self.rot_sym:
                                    if np.linalg.norm(v - axis) >= self.tol and np.dot(v, normal) < self.tol:
                                        mirror_type = 'v'
                                        break
                            else:
                                mirror_type = 'v'
                            break
        return mirror_type

    def _get_smallest_set_not_on_axis(self, axis):
        """Returns the smallest list of atoms with the same species and distance from
        origin AND does not lie on the specified axis.

        This maximal set limits the possible rotational symmetry operations, since atoms
        lying on a test axis is irrelevant in testing rotational symmetryOperations.
        """

        def not_on_axis(site):
            v = np.cross(site.coords, axis)
            return np.linalg.norm(v) > self.tol
        valid_sets = []
        _origin_site, dist_el_sites = cluster_sites(self.centered_mol, self.tol)
        for test_set in dist_el_sites.values():
            valid_set = list(filter(not_on_axis, test_set))
            if len(valid_set) > 0:
                valid_sets.append(valid_set)
        return min(valid_sets, key=len)

    def _check_rot_sym(self, axis):
        """Determines the rotational symmetry about supplied axis.

        Used only for symmetric top molecules which has possible rotational symmetry
        operations > 2.
        """
        min_set = self._get_smallest_set_not_on_axis(axis)
        max_sym = len(min_set)
        for i in range(max_sym, 0, -1):
            if max_sym % i != 0:
                continue
            op = SymmOp.from_axis_angle_and_translation(axis, 360 / i)
            rotvalid = self.is_valid_op(op)
            if rotvalid:
                self.symmops.append(op)
                self.rot_sym.append((axis, i))
                return i
        return 1

    def _check_perpendicular_r2_axis(self, axis):
        """Checks for R2 axes perpendicular to unique axis.

        For handling symmetric top molecules.
        """
        min_set = self._get_smallest_set_not_on_axis(axis)
        for s1, s2 in itertools.combinations(min_set, 2):
            test_axis = np.cross(s1.coords - s2.coords, axis)
            if np.linalg.norm(test_axis) > self.tol:
                op = SymmOp.from_axis_angle_and_translation(test_axis, 180)
                r2present = self.is_valid_op(op)
                if r2present:
                    self.symmops.append(op)
                    self.rot_sym.append((test_axis, 2))
                    return True
        return None

    def _proc_sph_top(self):
        """Handles Spherical Top Molecules, which belongs to the T, O or I point
        groups.
        """
        self._find_spherical_axes()
        if len(self.rot_sym) == 0:
            logger.debug('Accidental spherical top!')
            self._proc_sym_top()
        main_axis, rot = max(self.rot_sym, key=lambda v: v[1])
        if rot < 3:
            logger.debug('Accidental spherical top!')
            self._proc_sym_top()
        elif rot == 3:
            mirror_type = self._find_mirror(main_axis)
            if mirror_type != '':
                if self.is_valid_op(PointGroupAnalyzer.inversion_op):
                    self.symmops.append(PointGroupAnalyzer.inversion_op)
                    self.sch_symbol = 'Th'
                else:
                    self.sch_symbol = 'Td'
            else:
                self.sch_symbol = 'T'
        elif rot == 4:
            if self.is_valid_op(PointGroupAnalyzer.inversion_op):
                self.symmops.append(PointGroupAnalyzer.inversion_op)
                self.sch_symbol = 'Oh'
            else:
                self.sch_symbol = 'O'
        elif rot == 5:
            if self.is_valid_op(PointGroupAnalyzer.inversion_op):
                self.symmops.append(PointGroupAnalyzer.inversion_op)
                self.sch_symbol = 'Ih'
            else:
                self.sch_symbol = 'I'

    def _find_spherical_axes(self):
        """Looks for R5, R4, R3 and R2 axes in spherical top molecules.

        Point group T molecules have only one unique 3-fold and one unique 2-fold axis. O
        molecules have one unique 4, 3 and 2-fold axes. I molecules have a unique 5-fold
        axis.
        """
        rot_present = defaultdict(bool)
        _origin_site, dist_el_sites = cluster_sites(self.centered_mol, self.tol)
        test_set = min(dist_el_sites.values(), key=len)
        coords = [s.coords for s in test_set]
        for c1, c2, c3 in itertools.combinations(coords, 3):
            for cc1, cc2 in itertools.combinations([c1, c2, c3], 2):
                if not rot_present[2]:
                    test_axis = cc1 + cc2
                    if np.linalg.norm(test_axis) > self.tol:
                        op = SymmOp.from_axis_angle_and_translation(test_axis, 180)
                        rot_present[2] = self.is_valid_op(op)
                        if rot_present[2]:
                            self.symmops.append(op)
                            self.rot_sym.append((test_axis, 2))
            test_axis = np.cross(c2 - c1, c3 - c1)
            if np.linalg.norm(test_axis) > self.tol:
                for r in (3, 4, 5):
                    if not rot_present[r]:
                        op = SymmOp.from_axis_angle_and_translation(test_axis, 360 / r)
                        rot_present[r] = self.is_valid_op(op)
                        if rot_present[r]:
                            self.symmops.append(op)
                            self.rot_sym.append((test_axis, r))
                            break
            if rot_present[2] and rot_present[3] and (rot_present[4] or rot_present[5]):
                break

    def get_pointgroup(self):
        """Returns a PointGroup object for the molecule."""
        return PointGroupOperations(self.sch_symbol, self.symmops, self.mat_tol)

    def get_symmetry_operations(self):
        """Return symmetry operations as a list of SymmOp objects. Returns Cartesian coord
        symmops.

        Returns:
            list[SymmOp]: symmetry operations.
        """
        return generate_full_symmops(self.symmops, self.tol)

    def get_rotational_symmetry_number(self):
        """Return the rotational symmetry number."""
        symm_ops = self.get_symmetry_operations()
        symm_number = 0
        for symm in symm_ops:
            rot = symm.rotation_matrix
            if np.abs(np.linalg.det(rot) - 1) < 0.0001:
                symm_number += 1
        return symm_number

    def is_valid_op(self, symmop) -> bool:
        """Check if a particular symmetry operation is a valid symmetry operation for a
        molecule, i.e., the operation maps all atoms to another equivalent atom.

        Args:
            symmop (SymmOp): Symmetry operation to test.

        Returns:
            bool: Whether SymmOp is valid for Molecule.
        """
        coords = self.centered_mol.cart_coords
        for site in self.centered_mol:
            coord = symmop.operate(site.coords)
            ind = find_in_coord_list(coords, coord, self.tol)
            if not (len(ind) == 1 and self.centered_mol[ind[0]].species == site.species):
                return False
        return True

    def _get_eq_sets(self):
        """Calculates the dictionary for mapping equivalent atoms onto each other.

        Returns:
            dict: with two possible keys:
                eq_sets: A dictionary of indices mapping to sets of indices, each key maps to
                    indices of all equivalent atoms. The keys are guaranteed to be not equivalent.
                sym_ops: Twofold nested dictionary. operations[i][j] gives the symmetry
                    operation that maps atom i unto j.
        """
        UNIT = np.eye(3)
        eq_sets, operations = (defaultdict(set), defaultdict(dict))
        symm_ops = [op.rotation_matrix for op in generate_full_symmops(self.symmops, self.tol)]

        def get_clustered_indices():
            indices = cluster_sites(self.centered_mol, self.tol, give_only_index=True)
            out = list(indices[1].values())
            if indices[0] is not None:
                out.append([indices[0]])
            return out
        for index in get_clustered_indices():
            sites = self.centered_mol.cart_coords[index]
            for i, reference in zip(index, sites):
                for op in symm_ops:
                    rotated = np.dot(op, sites.T).T
                    matched_indices = find_in_coord_list(rotated, reference, self.tol)
                    matched_indices = {dict(enumerate(index))[i] for i in matched_indices}
                    eq_sets[i] |= matched_indices
                    if i not in operations:
                        operations[i] = {j: op.T if j != i else UNIT for j in matched_indices}
                    else:
                        for j in matched_indices:
                            if j not in operations[i]:
                                operations[i][j] = op.T if j != i else UNIT
                    for j in matched_indices:
                        if j not in operations:
                            operations[j] = {i: op if j != i else UNIT}
                        elif i not in operations[j]:
                            operations[j][i] = op if j != i else UNIT
        return {'eq_sets': eq_sets, 'sym_ops': operations}

    @staticmethod
    def _combine_eq_sets(equiv_sets, sym_ops):
        """Combines the dicts of _get_equivalent_atom_dicts into one.

        Args:
            equiv_sets (dict): Map of equivalent atoms onto each other (i.e. indices to indices).
            sym_ops (dict): Map of symmetry operations that map atoms onto each other.

        Returns:
            dict: with two possible keys:
                eq_sets: A dictionary of indices mapping to sets of indices, each key maps to
                    indices of all equivalent atoms. The keys are guaranteed to be not equivalent.
                sym_ops: Twofold nested dictionary. operations[i][j] gives the symmetry
                    operation that maps atom i unto j.
        """
        unit_mat = np.eye(3)

        def all_equivalent_atoms_of_i(idx, eq_sets, ops):
            """WORKS INPLACE on operations."""
            visited = {idx}
            tmp_eq_sets = {j: eq_sets[j] - visited for j in eq_sets[idx]}
            while tmp_eq_sets:
                new_tmp_eq_sets = {}
                for j in tmp_eq_sets:
                    if j in visited:
                        continue
                    visited.add(j)
                    for k in tmp_eq_sets[j]:
                        new_tmp_eq_sets[k] = eq_sets[k] - visited
                        if idx not in ops[k]:
                            ops[k][idx] = np.dot(ops[j][idx], ops[k][j]) if k != idx else unit_mat
                        ops[idx][k] = ops[k][idx].T
                tmp_eq_sets = new_tmp_eq_sets
            return (visited, ops)
        equiv_sets = copy.deepcopy(equiv_sets)
        ops = copy.deepcopy(sym_ops)
        to_be_deleted = set()
        for idx in equiv_sets:
            if idx in to_be_deleted:
                continue
            visited, ops = all_equivalent_atoms_of_i(idx, equiv_sets, ops)
            to_be_deleted |= visited - {idx}
        for key in to_be_deleted:
            equiv_sets.pop(key, None)
        return {'eq_sets': equiv_sets, 'sym_ops': ops}

    def get_equivalent_atoms(self):
        """Returns sets of equivalent atoms with symmetry operations.

        Returns:
            dict: with two possible keys:
                eq_sets: A dictionary of indices mapping to sets of indices, each key maps to
                    indices of all equivalent atoms. The keys are guaranteed to be not equivalent.
                sym_ops: Twofold nested dictionary. operations[i][j] gives the symmetry
                    operation that maps atom i unto j.
        """
        eq = self._get_eq_sets()
        return self._combine_eq_sets(eq['eq_sets'], eq['sym_ops'])

    def symmetrize_molecule(self):
        """Returns a symmetrized molecule.

        The equivalent atoms obtained via
        :meth:`~pymatgen.symmetry.analyzer.PointGroupAnalyzer.get_equivalent_atoms`
        are rotated, mirrored... unto one position.
        Then the average position is calculated.
        The average position is rotated, mirrored... back with the inverse
        of the previous symmetry operations, which gives the
        symmetrized molecule

        Returns:
            dict: with three possible keys:
                sym_mol: A symmetrized molecule instance.
                eq_sets: A dictionary of indices mapping to sets of indices, each key maps to indices
                    of all equivalent atoms. The keys are guaranteed to be not equivalent.
                sym_ops: Twofold nested dictionary. operations[i][j] gives the symmetry operation
                    that maps atom i unto j.
        """
        eq = self.get_equivalent_atoms()
        eq_sets, ops = (eq['eq_sets'], eq['sym_ops'])
        coords = self.centered_mol.cart_coords.copy()
        for i, eq_indices in eq_sets.items():
            for j in eq_indices:
                coords[j] = np.dot(ops[j][i], coords[j])
            coords[i] = np.mean(coords[list(eq_indices)], axis=0)
            for j in eq_indices:
                if j == i:
                    continue
                coords[j] = np.dot(ops[i][j], coords[i])
                coords[j] = np.dot(ops[i][j], coords[i])
        molecule = Molecule(species=self.centered_mol.species_and_occu, coords=coords)
        return {'sym_mol': molecule, 'eq_sets': eq_sets, 'sym_ops': ops}