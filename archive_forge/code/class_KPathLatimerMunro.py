from __future__ import annotations
import abc
import itertools
from math import ceil, cos, e, pi, sin, tan
from typing import TYPE_CHECKING, Any
from warnings import warn
import networkx as nx
import numpy as np
import spglib
from monty.dev import requires
from scipy.linalg import sqrtm
from pymatgen.core.lattice import Lattice
from pymatgen.core.operations import MagSymmOp, SymmOp
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer, cite_conventional_cell_algo
class KPathLatimerMunro(KPathBase):
    """This class looks for a path along high-symmetry lines in the
    Brillouin zone. It is based on the method outlined in:
    npj Comput Mater 6, 112 (2020). 10.1038/s41524-020-00383-7
    The user should ensure that the unit cell of the input structure
    is as reduced as possible, i.e. that there is no linear
    combination of lattice vectors which can produce a vector of
    lesser magnitude than the given set (this is required to
    obtain the correct Brillouin zone within the current
    implementation). This is checked during initialization and a
    warning is issued if the condition is not fulfilled.
    In the case of magnetic structures, care must also be taken to
    provide the magnetic primitive cell (i.e. that which reproduces
    the entire crystal, including the correct magnetic ordering,
    upon application of lattice translations). There is no algorithm to
        check for this, so if the input structure is
    incorrect, the class will output the incorrect k-path without
    any warning being issued.
    """

    def __init__(self, structure, has_magmoms=False, magmom_axis=None, symprec=0.01, angle_tolerance=5, atol=1e-05):
        """
        Args:
            structure (Structure): Structure object
            has_magmoms (bool): Whether the input structure contains
                magnetic moments as site properties with the key 'magmom.'
                Values may be in the form of 3-component vectors given in
                the basis of the input lattice vectors, or as scalars, in
                which case the spin axis will default to a_3, the third
                real-space lattice vector (this triggers a warning).
            magmom_axis (list or numpy array): 3-component vector specifying
                direction along which magnetic moments given as scalars
                should point. If all magnetic moments are provided as
                vectors then this argument is not used.
            symprec (float): Tolerance for symmetry finding
            angle_tolerance (float): Angle tolerance for symmetry finding.
            atol (float): Absolute tolerance used to determine symmetric
                equivalence of points and lines in the BZ.
        """
        super().__init__(structure, symprec=symprec, angle_tolerance=angle_tolerance, atol=atol)
        reducible = []
        for i in range(3):
            for j in range(3):
                if i != j:
                    if np.absolute(np.dot(self._latt.matrix[i], self._latt.matrix[j])) > np.dot(self._latt.matrix[i], self._latt.matrix[i]) and np.absolute(np.dot(self._latt.matrix[i], self._latt.matrix[j]) - np.dot(self._latt.matrix[i], self._latt.matrix[i])) > atol:
                        reducible.append(True)
                    else:
                        reducible.append(False)
        if np.any(reducible):
            print('reducible')
            warn('The unit cell of the input structure is not fully reduced!The path may be incorrect. Use at your own risk.')
        if magmom_axis is None:
            magmom_axis = np.array([0, 0, 1])
            axis_specified = False
        else:
            axis_specified = True
        self._kpath = self._get_ksymm_kpath(has_magmoms, magmom_axis, axis_specified, symprec, angle_tolerance, atol)

    @property
    def mag_type(self):
        """
        Returns:
            The type of magnetic space group as a string. Current implementation does not
            distinguish between types 3 and 4, so return value is '3/4'. If has_magmoms is
            False, returns '0'.
        """
        return self._mag_type

    def _get_ksymm_kpath(self, has_magmoms, magmom_axis, axis_specified, symprec, angle_tolerance, atol):
        ID = np.eye(3)
        PAR = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        V = self._latt.matrix.T
        W = self._rec_lattice.matrix.T
        A = np.dot(np.linalg.inv(W), V)
        if has_magmoms:
            grey_struct = self._structure.copy()
            grey_struct.remove_site_property('magmom')
            sga = SpacegroupAnalyzer(grey_struct, symprec=symprec, angle_tolerance=angle_tolerance)
            grey_ops = sga.get_symmetry_operations()
            self._structure = self._convert_all_magmoms_to_vectors(magmom_axis, axis_specified)
            mag_ops = self._get_magnetic_symmetry_operations(self._structure, grey_ops, atol)
            D = [SymmOp.from_rotation_and_translation(rotation_matrix=op.rotation_matrix, translation_vec=op.translation_vector) for op in mag_ops if op.time_reversal == 1]
            fD = [SymmOp.from_rotation_and_translation(rotation_matrix=op.rotation_matrix, translation_vec=op.translation_vector) for op in mag_ops if op.time_reversal == -1]
            if np.array([m == np.array([0, 0, 0]) for m in self._structure.site_properties['magmom']]).all():
                fD = D
                D = []
            if len(fD) == 0:
                self._mag_type = '1'
                isomorphic_point_group = [d.rotation_matrix for d in D]
                recip_point_group = self._get_reciprocal_point_group(isomorphic_point_group, ID, A)
            elif len(D) == 0:
                self._mag_type = '2'
                isomorphic_point_group = [d.rotation_matrix for d in fD]
                recip_point_group = self._get_reciprocal_point_group(isomorphic_point_group, PAR, A)
            else:
                self._mag_type = '3/4'
                f = self._get_coset_factor(D + fD, D)
                isomorphic_point_group = [d.rotation_matrix for d in D]
                recip_point_group = self._get_reciprocal_point_group(isomorphic_point_group, np.dot(PAR, f.rotation_matrix), A)
        else:
            self._mag_type = '0'
            if 'magmom' in self._structure.site_properties:
                warn('The parameter has_magmoms is False, but site_properties contains the key magmom.This property will be removed and could result in different symmetry operations.')
                self._structure.remove_site_property('magmom')
            sga = SpacegroupAnalyzer(self._structure)
            ops = sga.get_symmetry_operations()
            isomorphic_point_group = [op.rotation_matrix for op in ops]
            recip_point_group = self._get_reciprocal_point_group(isomorphic_point_group, PAR, A)
        self._rpg = recip_point_group
        key_points, bz_as_key_point_inds, face_center_inds = self._get_key_points()
        key_points_inds_orbits = self._get_key_point_orbits(key_points=key_points)
        key_lines = self._get_key_lines(key_points=key_points, bz_as_key_point_inds=bz_as_key_point_inds)
        key_lines_inds_orbits = self._get_key_line_orbits(key_points=key_points, key_lines=key_lines, key_points_inds_orbits=key_points_inds_orbits)
        little_groups_points, little_groups_lines = self._get_little_groups(key_points=key_points, key_points_inds_orbits=key_points_inds_orbits, key_lines_inds_orbits=key_lines_inds_orbits)
        _point_orbits_in_path, line_orbits_in_path = self._choose_path(key_points=key_points, key_points_inds_orbits=key_points_inds_orbits, key_lines_inds_orbits=key_lines_inds_orbits, little_groups_points=little_groups_points, little_groups_lines=little_groups_lines)
        IRBZ_points_inds = self._get_IRBZ(recip_point_group, W, key_points, face_center_inds, atol)
        lines_in_path_inds = []
        for ind in line_orbits_in_path:
            for tup in key_lines_inds_orbits[ind]:
                if tup[0] in IRBZ_points_inds and tup[1] in IRBZ_points_inds:
                    lines_in_path_inds.append(tup)
                    break
        G = nx.Graph(lines_in_path_inds)
        lines_in_path_inds = list(nx.edge_dfs(G))
        points_in_path_inds = [ind for tup in lines_in_path_inds for ind in tup]
        points_in_path_inds_unique = list(set(points_in_path_inds))
        orbit_cosines = []
        for orbit in key_points_inds_orbits[:-1]:
            current_orbit_cosines = []
            for orbit_index in orbit:
                key_point = key_points[orbit_index]
                for point_index in range(26):
                    label_point = self.label_points(point_index)
                    cosine_value = np.dot(key_point, label_point)
                    cosine_value /= np.linalg.norm(key_point) * np.linalg.norm(label_point)
                    cosine_value = np.round(cosine_value, decimals=3)
                    current_orbit_cosines.append((point_index, cosine_value))
            sorted_cosines = sorted(current_orbit_cosines, key=lambda x: (-x[1], x[0]))
            orbit_cosines.append(sorted_cosines)
        orbit_labels = self._get_orbit_labels(orbit_cosines, key_points_inds_orbits, atol)
        key_points_labels = ['' for i in range(len(key_points))]
        for idx, orbit in enumerate(key_points_inds_orbits):
            for point_ind in orbit:
                key_points_labels[point_ind] = self.label_symbol(int(orbit_labels[idx]))
        kpoints = {}
        reverse_kpoints = {}
        for point_ind in points_in_path_inds_unique:
            point_label = key_points_labels[point_ind]
            if point_label not in kpoints:
                kpoints[point_label] = key_points[point_ind]
                reverse_kpoints[point_ind] = point_label
            else:
                existing_labels = [key for key in kpoints if point_label in key]
                if "'" not in point_label:
                    existing_labels[:] = [label for label in existing_labels if "'" not in label]
                if len(existing_labels) == 1:
                    max_occurence = 0
                elif "'" not in point_label:
                    max_occurence = max((int(label[3:-1]) for label in existing_labels[1:]))
                else:
                    max_occurence = max((int(label[4:-1]) for label in existing_labels[1:]))
                kpoints[f'{point_label}_{{{max_occurence + 1}}}'] = key_points[point_ind]
                reverse_kpoints[point_ind] = f'{point_label}_{{{max_occurence + 1}}}'
        path = []
        idx = 0
        start_of_subpath = True
        while idx < len(points_in_path_inds):
            if start_of_subpath:
                path.append([reverse_kpoints[points_in_path_inds[idx]]])
                idx += 1
                start_of_subpath = False
            elif points_in_path_inds[idx] == points_in_path_inds[idx + 1]:
                path[-1].append(reverse_kpoints[points_in_path_inds[idx]])
                idx += 2
            else:
                path[-1].append(reverse_kpoints[points_in_path_inds[idx]])
                idx += 1
                start_of_subpath = True
            if idx == len(points_in_path_inds) - 1:
                path[-1].append(reverse_kpoints[points_in_path_inds[idx]])
                idx += 1
        return {'kpoints': kpoints, 'path': path}

    def _choose_path(self, key_points, key_points_inds_orbits, key_lines_inds_orbits, little_groups_points, little_groups_lines):
        ID = np.eye(3)
        PAR = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        gamma_ind = len(key_points) - 1
        line_orbits_in_path = []
        point_orbits_in_path = []
        for idx, little_group in enumerate(little_groups_lines):
            add_rep = False
            nC2 = 0
            nC3 = 0
            nsig = 0
            for opind in little_group:
                op = self._rpg[opind]
                if not (op == ID).all():
                    if (np.dot(op, op) == ID).all():
                        if np.linalg.det(op) == 1:
                            nC2 += 1
                            break
                        if not (op == PAR).all():
                            nsig += 1
                            break
                    elif (np.dot(op, np.dot(op, op)) == ID).all():
                        nC3 += 1
                        break
            if nC2 > 0 or nC3 > 0 or nsig > 0:
                add_rep = True
            if add_rep:
                line_orbits_in_path.append(idx)
                line = key_lines_inds_orbits[idx][0]
                ind0 = line[0]
                ind1 = line[1]
                found0 = False
                found1 = False
                for j, orbit in enumerate(key_points_inds_orbits):
                    if ind0 in orbit:
                        point_orbits_in_path.append(j)
                        found0 = True
                    if ind1 in orbit:
                        point_orbits_in_path.append(j)
                        found1 = True
                    if found0 and found1:
                        break
        point_orbits_in_path = list(set(point_orbits_in_path))
        unconnected = []
        for idx in range(len(key_points_inds_orbits)):
            if idx not in point_orbits_in_path:
                unconnected.append(idx)
        for ind in unconnected:
            connect = False
            for op_ind in little_groups_points[ind]:
                op = self._rpg[op_ind]
                if (op == ID).all():
                    pass
                elif (op == PAR).all():
                    connect = True
                    break
                elif np.linalg.det(op) == 1:
                    if (np.dot(op, np.dot(op, op)) == ID).all():
                        pass
                    else:
                        connect = True
                        break
                else:
                    pass
            if connect:
                line = (key_points_inds_orbits[ind][0], gamma_ind)
                for j, orbit in enumerate(key_lines_inds_orbits):
                    if line in orbit:
                        line_orbits_in_path.append(j)
                        break
                if gamma_ind not in point_orbits_in_path:
                    point_orbits_in_path.append(gamma_ind)
                point_orbits_in_path.append(ind)
        return (point_orbits_in_path, line_orbits_in_path)

    def _get_key_points(self):
        decimals = ceil(-1 * np.log10(self._atol)) - 1
        bz = self._rec_lattice.get_wigner_seitz_cell()
        key_points = []
        face_center_inds = []
        bz_as_key_point_inds = []
        for idx, facet in enumerate(bz):
            for j, vert in enumerate(facet):
                vert = self._rec_lattice.get_fractional_coords(vert)
                bz[idx][j] = vert
        pop = []
        for idx, facet in enumerate(bz):
            rounded_facet = np.around(facet, decimals=decimals)
            u, indices = np.unique(rounded_facet, axis=0, return_index=True)
            if len(u) in [1, 2]:
                pop.append(idx)
            else:
                bz[idx] = [facet[j] for j in np.sort(indices)]
        bz = [bz[i] for i in range(len(bz)) if i not in pop]
        for idx, facet in enumerate(bz):
            bz_as_key_point_inds.append([])
            for j, vert in enumerate(facet):
                edge_center = (vert + facet[j + 1]) / 2 if j != len(facet) - 1 else (vert + facet[0]) / 2.0
                duplicatevert = False
                duplicateedge = False
                for k, point in enumerate(key_points):
                    if np.allclose(vert, point, atol=self._atol):
                        bz_as_key_point_inds[idx].append(k)
                        duplicatevert = True
                        break
                for k, point in enumerate(key_points):
                    if np.allclose(edge_center, point, atol=self._atol):
                        bz_as_key_point_inds[idx].append(k)
                        duplicateedge = True
                        break
                if not duplicatevert:
                    key_points.append(vert)
                    bz_as_key_point_inds[idx].append(len(key_points) - 1)
                if not duplicateedge:
                    key_points.append(edge_center)
                    bz_as_key_point_inds[idx].append(len(key_points) - 1)
            if len(facet) == 4:
                face_center = (facet[0] + facet[1] + facet[2] + facet[3]) / 4.0
                key_points.append(face_center)
                face_center_inds.append(len(key_points) - 1)
                bz_as_key_point_inds[idx].append(len(key_points) - 1)
            else:
                face_center = (facet[0] + facet[1] + facet[2] + facet[3] + facet[4] + facet[5]) / 6.0
                key_points.append(face_center)
                face_center_inds.append(len(key_points) - 1)
                bz_as_key_point_inds[idx].append(len(key_points) - 1)
        key_points.append(np.array([0, 0, 0]))
        return (key_points, bz_as_key_point_inds, face_center_inds)

    def _get_key_point_orbits(self, key_points):
        key_points_copy = dict(zip(range(len(key_points) - 1), key_points[0:len(key_points) - 1]))
        key_points_inds_orbits = []
        i = 0
        while len(key_points_copy) > 0:
            key_points_inds_orbits.append([])
            k0ind = next(iter(key_points_copy))
            k0 = key_points_copy[k0ind]
            key_points_inds_orbits[i].append(k0ind)
            key_points_copy.pop(k0ind)
            for op in self._rpg:
                to_pop = []
                k1 = np.dot(op, k0)
                for ind_key in key_points_copy:
                    diff = k1 - key_points_copy[ind_key]
                    if self._all_ints(diff, atol=self._atol):
                        key_points_inds_orbits[i].append(ind_key)
                        to_pop.append(ind_key)
                for key in to_pop:
                    key_points_copy.pop(key)
            i += 1
        key_points_inds_orbits.append([len(key_points) - 1])
        return key_points_inds_orbits

    @staticmethod
    def _get_key_lines(key_points, bz_as_key_point_inds):
        key_lines = []
        gamma_ind = len(key_points) - 1
        for facet_as_key_point_inds in bz_as_key_point_inds:
            facet_as_key_point_inds_bndy = facet_as_key_point_inds[:len(facet_as_key_point_inds) - 1]
            face_center_ind = facet_as_key_point_inds[-1]
            for j, ind in enumerate(facet_as_key_point_inds_bndy, start=-1):
                if (min(ind, facet_as_key_point_inds_bndy[j]), max(ind, facet_as_key_point_inds_bndy[j])) not in key_lines:
                    key_lines.append((min(ind, facet_as_key_point_inds_bndy[j]), max(ind, facet_as_key_point_inds_bndy[j])))
                k = j + 2 if j != len(facet_as_key_point_inds_bndy) - 2 else 0
                if (min(ind, facet_as_key_point_inds_bndy[k]), max(ind, facet_as_key_point_inds_bndy[k])) not in key_lines:
                    key_lines.append((min(ind, facet_as_key_point_inds_bndy[k]), max(ind, facet_as_key_point_inds_bndy[k])))
                if (ind, gamma_ind) not in key_lines:
                    key_lines.append((ind, gamma_ind))
                key_lines.append((min(ind, face_center_ind), max(ind, face_center_ind)))
            key_lines.append((face_center_ind, gamma_ind))
        return key_lines

    def _get_key_line_orbits(self, key_points, key_lines, key_points_inds_orbits):
        key_lines_copy = dict(zip(range(len(key_lines)), key_lines))
        key_lines_inds_orbits = []
        i = 0
        while len(key_lines_copy) > 0:
            key_lines_inds_orbits.append([])
            l0ind = next(iter(key_lines_copy))
            l0 = key_lines_copy[l0ind]
            key_lines_inds_orbits[i].append(l0)
            key_lines_copy.pop(l0ind)
            to_pop = []
            p00 = key_points[l0[0]]
            p01 = key_points[l0[1]]
            pmid0 = p00 + e / pi * (p01 - p00)
            for ind_key in key_lines_copy:
                l1 = key_lines_copy[ind_key]
                p10 = key_points[l1[0]]
                p11 = key_points[l1[1]]
                equivptspar = False
                equivptsperp = False
                equivline = False
                if np.array([l0[0] in orbit and l1[0] in orbit for orbit in key_points_inds_orbits]).any() and np.array([l0[1] in orbit and l1[1] in orbit for orbit in key_points_inds_orbits]).any():
                    equivptspar = True
                elif np.array([l0[1] in orbit and l1[0] in orbit for orbit in key_points_inds_orbits]).any() and np.array([l0[0] in orbit and l1[1] in orbit for orbit in key_points_inds_orbits]).any():
                    equivptsperp = True
                if equivptspar:
                    pmid1 = p10 + e / pi * (p11 - p10)
                    for op in self._rpg:
                        if not equivline:
                            p00pr = np.dot(op, p00)
                            diff0 = p10 - p00pr
                            if self._all_ints(diff0, atol=self._atol):
                                pmid0pr = np.dot(op, pmid0) + diff0
                                p01pr = np.dot(op, p01) + diff0
                                if np.allclose(p11, p01pr, atol=self._atol) and np.allclose(pmid1, pmid0pr, atol=self._atol):
                                    equivline = True
                elif equivptsperp:
                    pmid1 = p11 + e / pi * (p10 - p11)
                    for op in self._rpg:
                        if not equivline:
                            p00pr = np.dot(op, p00)
                            diff0 = p11 - p00pr
                            if self._all_ints(diff0, atol=self._atol):
                                pmid0pr = np.dot(op, pmid0) + diff0
                                p01pr = np.dot(op, p01) + diff0
                                if np.allclose(p10, p01pr, atol=self._atol) and np.allclose(pmid1, pmid0pr, atol=self._atol):
                                    equivline = True
                if equivline:
                    key_lines_inds_orbits[i].append(l1)
                    to_pop.append(ind_key)
            for key in to_pop:
                key_lines_copy.pop(key)
            i += 1
        return key_lines_inds_orbits

    def _get_little_groups(self, key_points, key_points_inds_orbits, key_lines_inds_orbits):
        little_groups_points = []
        for i, orbit in enumerate(key_points_inds_orbits):
            k0 = key_points[orbit[0]]
            little_groups_points.append([])
            for j, op in enumerate(self._rpg):
                gamma_to = np.dot(op, -1 * k0) + k0
                check_gamma = True
                if not self._all_ints(gamma_to, atol=self._atol):
                    check_gamma = False
                if check_gamma:
                    little_groups_points[i].append(j)
        little_groups_lines = []
        for i, orbit in enumerate(key_lines_inds_orbits):
            l0 = orbit[0]
            v = key_points[l0[1]] - key_points[l0[0]]
            k0 = key_points[l0[0]] + np.e / pi * v
            little_groups_lines.append([])
            for j, op in enumerate(self._rpg):
                gamma_to = np.dot(op, -1 * k0) + k0
                check_gamma = True
                if not self._all_ints(gamma_to, atol=self._atol):
                    check_gamma = False
                if check_gamma:
                    little_groups_lines[i].append(j)
        return (little_groups_points, little_groups_lines)

    def _convert_all_magmoms_to_vectors(self, magmom_axis, axis_specified):
        struct = self._structure.copy()
        magmom_axis = np.array(magmom_axis)
        if 'magmom' not in struct.site_properties:
            warn("The 'magmom' property is not set in the structure's site properties.All magnetic moments are being set to zero.")
            struct.add_site_property('magmom', [np.array([0, 0, 0]) for _ in range(len(struct))])
            return struct
        old_magmoms = struct.site_properties['magmom']
        new_magmoms = []
        found_scalar = False
        for magmom in old_magmoms:
            if isinstance(magmom, np.ndarray):
                new_magmoms.append(magmom)
            elif isinstance(magmom, list):
                new_magmoms.append(np.array(magmom))
            else:
                found_scalar = True
                new_magmoms.append(magmom * magmom_axis)
        if found_scalar and (not axis_specified):
            warn('At least one magmom had a scalar value and magmom_axis was not specified. Defaulted to z+ spinor.')
        struct.remove_site_property('magmom')
        struct.add_site_property('magmom', new_magmoms)
        return struct

    def _get_magnetic_symmetry_operations(self, struct, grey_ops, atol):
        mag_ops = []
        magmoms = struct.site_properties['magmom']
        nonzero_magmom_inds = [idx for idx in range(len(struct)) if not (magmoms[idx] == np.array([0, 0, 0])).all()]
        init_magmoms = [site.properties['magmom'] for idx, site in enumerate(struct) if idx in nonzero_magmom_inds]
        sites = [site for idx, site in enumerate(struct) if idx in nonzero_magmom_inds]
        init_site_coords = [site.frac_coords for site in sites]
        for op in grey_ops:
            rot_mat = op.rotation_matrix
            t = op.translation_vector
            xformed_magmoms = [self._apply_op_to_magmom(rot_mat, magmom) for magmom in init_magmoms]
            xformed_site_coords = [np.dot(rot_mat, site.frac_coords) + t for site in sites]
            permutation = ['a' for i in range(len(sites))]
            not_found = list(range(len(sites)))
            for i in range(len(sites)):
                xformed = xformed_site_coords[i]
                for k, j in enumerate(not_found):
                    init = init_site_coords[j]
                    diff = xformed - init
                    if self._all_ints(diff, atol=atol):
                        permutation[i] = j
                        not_found.pop(k)
                        break
            same = np.zeros(len(sites))
            flipped = np.zeros(len(sites))
            for i, magmom in enumerate(xformed_magmoms):
                if (magmom == init_magmoms[permutation[i]]).all():
                    same[i] = 1
                elif (magmom == -1 * init_magmoms[permutation[i]]).all():
                    flipped[i] = 1
            if same.all():
                mag_ops.append(MagSymmOp.from_rotation_and_translation_and_time_reversal(rotation_matrix=op.rotation_matrix, translation_vec=op.translation_vector, time_reversal=1))
            if flipped.all():
                mag_ops.append(MagSymmOp.from_rotation_and_translation_and_time_reversal(rotation_matrix=op.rotation_matrix, translation_vec=op.translation_vector, time_reversal=-1))
        return mag_ops

    @staticmethod
    def _get_reciprocal_point_group(ops, R, A):
        A_inv = np.linalg.inv(A)
        recip_point_group = [np.around(np.dot(A, np.dot(R, A_inv)), decimals=2)]
        for op in ops:
            recip = np.around(np.dot(A, np.dot(op, A_inv)), decimals=2)
            new = True
            new_coset = True
            for thing in recip_point_group:
                if (thing == recip).all():
                    new = False
                if (thing == np.dot(R, recip)).all():
                    new_coset = False
            if new:
                recip_point_group.append(recip)
            if new_coset:
                recip_point_group.append(np.dot(R, recip))
        return recip_point_group

    @staticmethod
    def _closewrapped(pos1, pos2, tolerance):
        pos1 = pos1 % 1.0
        pos2 = pos2 % 1.0
        if len(pos1) != len(pos2):
            return False
        for idx, p1 in enumerate(pos1):
            if abs(p1 - pos2[idx]) > tolerance[idx] and abs(p1 - pos2[idx]) < 1 - tolerance[idx]:
                return False
        return True

    def _get_coset_factor(self, G, H):
        gH = []
        for op1 in G:
            in_H = False
            for op2 in H:
                if np.allclose(op1.rotation_matrix, op2.rotation_matrix, atol=self._atol) and self._closewrapped(op1.translation_vector, op2.translation_vector, np.ones(3) * self._atol):
                    in_H = True
                    break
            if not in_H:
                gH.append(op1)
        for op in gH:
            opH = [op * h for h in H]
            is_coset_factor = True
            for op1 in opH:
                for op2 in H:
                    if np.allclose(op1.rotation_matrix, op2.rotation_matrix, atol=self._atol) and self._closewrapped(op1.translation_vector, op2.translation_vector, np.ones(3) * self._atol):
                        is_coset_factor = False
                        break
                if not is_coset_factor:
                    break
            if is_coset_factor:
                return op
        return 'No coset factor found.'

    @staticmethod
    def _apply_op_to_magmom(r, magmom):
        if np.linalg.det(r) == 1:
            return np.dot(r, magmom)
        return -1 * np.dot(r, magmom)

    @staticmethod
    def _all_ints(arr, atol):
        rounded_arr = np.around(arr, decimals=0)
        return np.allclose(rounded_arr, arr, atol=atol)

    def _get_IRBZ(self, recip_point_group, W, key_points, face_center_inds, atol):
        rpgdict = self._get_reciprocal_point_group_dict(recip_point_group, atol)
        g = np.dot(W.T, W)
        g_inv = np.linalg.inv(g)
        D = np.linalg.det(W)
        primary_orientation = secondary_orientation = tertiary_orientation = None
        planar_boundaries = []
        IRBZ_points = list(enumerate(key_points))
        for sigma in rpgdict['reflections']:
            norm = sigma['normal']
            if primary_orientation is None:
                primary_orientation = norm
                planar_boundaries.append(norm)
            elif np.isclose(np.dot(primary_orientation, np.dot(g, norm)), 0, atol=atol):
                if secondary_orientation is None:
                    secondary_orientation = norm
                    planar_boundaries.append(norm)
                elif np.isclose(np.dot(secondary_orientation, np.dot(g, norm)), 0, atol=atol):
                    if tertiary_orientation is None:
                        tertiary_orientation = norm
                        planar_boundaries.append(norm)
                    elif np.allclose(norm, -1 * tertiary_orientation, atol=atol):
                        pass
                elif np.dot(secondary_orientation, np.dot(g, norm)) < 0:
                    planar_boundaries.append(-1 * norm)
                else:
                    planar_boundaries.append(norm)
            elif np.dot(primary_orientation, np.dot(g, norm)) < 0:
                planar_boundaries.append(-1 * norm)
            else:
                planar_boundaries.append(norm)
        IRBZ_points = self._reduce_IRBZ(IRBZ_points, planar_boundaries, g, atol)
        used_axes = []
        for rotn in rpgdict['rotations']['six-fold']:
            ax = rotn['axis']
            op = rotn['op']
            if not np.any([np.allclose(ax, usedax, atol) for usedax in used_axes]) and self._op_maps_IRBZ_to_self(op, IRBZ_points, atol):
                face_center_found = False
                for point in IRBZ_points:
                    if point[0] in face_center_inds:
                        cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                        if not np.allclose(cross, 0, atol=atol):
                            rot_boundaries = [cross, -1 * np.dot(op, cross)]
                            face_center_found = True
                            used_axes.append(ax)
                            break
                if not face_center_found:
                    print('face center not found')
                    for point in IRBZ_points:
                        cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                        if not np.allclose(cross, 0, atol=atol):
                            rot_boundaries = [cross, -1 * np.dot(op, cross)]
                            used_axes.append(ax)
                            break
                IRBZ_points = self._reduce_IRBZ(IRBZ_points, rot_boundaries, g, atol)
        for rotn in rpgdict['rotations']['rotoinv-four-fold']:
            ax = rotn['axis']
            op = rotn['op']
            if not np.any([np.allclose(ax, usedax, atol) for usedax in used_axes]) and self._op_maps_IRBZ_to_self(op, IRBZ_points, atol):
                face_center_found = False
                for point in IRBZ_points:
                    if point[0] in face_center_inds:
                        cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                        if not np.allclose(cross, 0, atol=atol):
                            rot_boundaries = [cross, np.dot(op, cross)]
                            face_center_found = True
                            used_axes.append(ax)
                            break
                if not face_center_found:
                    print('face center not found')
                    for point in IRBZ_points:
                        cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                        if not np.allclose(cross, 0, atol=atol):
                            rot_boundaries = [cross, -1 * np.dot(op, cross)]
                            used_axes.append(ax)
                            break
                IRBZ_points = self._reduce_IRBZ(IRBZ_points, rot_boundaries, g, atol)
        for rotn in rpgdict['rotations']['four-fold']:
            ax = rotn['axis']
            op = rotn['op']
            if not np.any([np.allclose(ax, usedax, atol) for usedax in used_axes]) and self._op_maps_IRBZ_to_self(op, IRBZ_points, atol):
                face_center_found = False
                for point in IRBZ_points:
                    if point[0] in face_center_inds:
                        cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                        if not np.allclose(cross, 0, atol=atol):
                            rot_boundaries = [cross, -1 * np.dot(op, cross)]
                            face_center_found = True
                            used_axes.append(ax)
                            break
                if not face_center_found:
                    print('face center not found')
                    for point in IRBZ_points:
                        cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                        if not np.allclose(cross, 0, atol=atol):
                            rot_boundaries = [cross, -1 * np.dot(op, cross)]
                            used_axes.append(ax)
                            break
                IRBZ_points = self._reduce_IRBZ(IRBZ_points, rot_boundaries, g, atol)
        for rotn in rpgdict['rotations']['rotoinv-three-fold']:
            ax = rotn['axis']
            op = rotn['op']
            if not np.any([np.allclose(ax, usedax, atol) for usedax in used_axes]) and self._op_maps_IRBZ_to_self(op, IRBZ_points, atol):
                face_center_found = False
                for point in IRBZ_points:
                    if point[0] in face_center_inds:
                        cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                        if not np.allclose(cross, 0, atol=atol):
                            rot_boundaries = [cross, -1 * np.dot(sqrtm(-1 * op), cross)]
                            face_center_found = True
                            used_axes.append(ax)
                            break
                if not face_center_found:
                    print('face center not found')
                    for point in IRBZ_points:
                        cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                        if not np.allclose(cross, 0, atol=atol):
                            rot_boundaries = [cross, -1 * np.dot(op, cross)]
                            used_axes.append(ax)
                            break
                IRBZ_points = self._reduce_IRBZ(IRBZ_points, rot_boundaries, g, atol)
        for rotn in rpgdict['rotations']['three-fold']:
            ax = rotn['axis']
            op = rotn['op']
            if not np.any([np.allclose(ax, usedax, atol) for usedax in used_axes]) and self._op_maps_IRBZ_to_self(op, IRBZ_points, atol):
                face_center_found = False
                for point in IRBZ_points:
                    if point[0] in face_center_inds:
                        cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                        if not np.allclose(cross, 0, atol=atol):
                            rot_boundaries = [cross, -1 * np.dot(op, cross)]
                            face_center_found = True
                            used_axes.append(ax)
                            break
                if not face_center_found:
                    print('face center not found')
                    for point in IRBZ_points:
                        cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                        if not np.allclose(cross, 0, atol=atol):
                            rot_boundaries = [cross, -1 * np.dot(op, cross)]
                            used_axes.append(ax)
                            break
                IRBZ_points = self._reduce_IRBZ(IRBZ_points, rot_boundaries, g, atol)
        for rotn in rpgdict['rotations']['two-fold']:
            ax = rotn['axis']
            op = rotn['op']
            if not np.any([np.allclose(ax, usedax, atol) for usedax in used_axes]) and self._op_maps_IRBZ_to_self(op, IRBZ_points, atol):
                face_center_found = False
                for point in IRBZ_points:
                    if point[0] in face_center_inds:
                        cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                        if not np.allclose(cross, 0, atol=atol):
                            rot_boundaries = [cross, -1 * np.dot(op, cross)]
                            face_center_found = True
                            used_axes.append(ax)
                            break
                if not face_center_found:
                    print('face center not found')
                    for point in IRBZ_points:
                        cross = D * np.dot(g_inv, np.cross(ax, point[1]))
                        if not np.allclose(cross, 0, atol=atol):
                            rot_boundaries = [cross, -1 * np.dot(op, cross)]
                            used_axes.append(ax)
                            break
                IRBZ_points = self._reduce_IRBZ(IRBZ_points, rot_boundaries, g, atol)
        return [point[0] for point in IRBZ_points]

    @staticmethod
    def _get_reciprocal_point_group_dict(recip_point_group, atol):
        PAR = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])
        dct = {'reflections': [], 'rotations': {'two-fold': [], 'three-fold': [], 'four-fold': [], 'six-fold': [], 'rotoinv-three-fold': [], 'rotoinv-four-fold': [], 'rotoinv-six-fold': []}, 'inversion': []}
        for idx, op in enumerate(recip_point_group):
            evals, evects = np.linalg.eig(op)
            tr = np.trace(op)
            det = np.linalg.det(op)
            if np.isclose(det, 1, atol=atol):
                if np.isclose(tr, 3, atol=atol):
                    continue
                if np.isclose(tr, -1, atol=atol):
                    for j in range(3):
                        if np.isclose(evals[j], 1, atol=atol):
                            ax = evects[:, j]
                    dct['rotations']['two-fold'].append({'ind': idx, 'axis': ax, 'op': op})
                elif np.isclose(tr, 0, atol=atol):
                    for j in range(3):
                        if np.isreal(evals[j]) and np.isclose(np.absolute(evals[j]), 1, atol=atol):
                            ax = evects[:, j]
                    dct['rotations']['three-fold'].append({'ind': idx, 'axis': ax, 'op': op})
                elif np.isclose(tr, 1, atol=atol):
                    for j in range(3):
                        if np.isreal(evals[j]) and np.isclose(np.absolute(evals[j]), 1, atol=atol):
                            ax = evects[:, j]
                    dct['rotations']['four-fold'].append({'ind': idx, 'axis': ax, 'op': op})
                elif np.isclose(tr, 2, atol=atol):
                    for j in range(3):
                        if np.isreal(evals[j]) and np.isclose(np.absolute(evals[j]), 1, atol=atol):
                            ax = evects[:, j]
                    dct['rotations']['six-fold'].append({'ind': idx, 'axis': ax, 'op': op})
            if np.isclose(det, -1, atol=atol):
                if np.isclose(tr, -3, atol=atol):
                    dct['inversion'].append({'ind': idx, 'op': PAR})
                elif np.isclose(tr, 1, atol=atol):
                    for j in range(3):
                        if np.isclose(evals[j], -1, atol=atol):
                            norm = evects[:, j]
                    dct['reflections'].append({'ind': idx, 'normal': norm, 'op': op})
                elif np.isclose(tr, 0, atol=atol):
                    for j in range(3):
                        if np.isreal(evals[j]) and np.isclose(np.absolute(evals[j]), 1, atol=atol):
                            ax = evects[:, j]
                    dct['rotations']['rotoinv-three-fold'].append({'ind': idx, 'axis': ax, 'op': op})
                elif np.isclose(tr, -1, atol=atol):
                    for j in range(3):
                        if np.isreal(evals[j]) and np.isclose(np.absolute(evals[j]), 1, atol=atol):
                            ax = evects[:, j]
                    dct['rotations']['rotoinv-four-fold'].append({'ind': idx, 'axis': ax, 'op': op})
                elif np.isclose(tr, -2, atol=atol):
                    for j in range(3):
                        if np.isreal(evals[j]) and np.isclose(np.absolute(evals[j]), 1, atol=atol):
                            ax = evects[:, j]
                    dct['rotations']['rotoinv-six-fold'].append({'ind': idx, 'axis': ax, 'op': op})
        return dct

    @staticmethod
    def _op_maps_IRBZ_to_self(op, IRBZ_points, atol):
        point_coords = [point[1] for point in IRBZ_points]
        for point in point_coords:
            point_prime = np.dot(op, point)
            mapped_back = False
            for checkpoint in point_coords:
                if np.allclose(point_prime, checkpoint, atol):
                    mapped_back = True
                    break
            if not mapped_back:
                return False
        return True

    @staticmethod
    def _reduce_IRBZ(IRBZ_points, boundaries, g, atol):
        in_reduced_section = []
        for point in IRBZ_points:
            in_reduced_section.append(np.all([np.dot(point[1], np.dot(g, boundary)) >= 0 or np.isclose(np.dot(point[1], np.dot(g, boundary)), 0, atol=atol) for boundary in boundaries]))
        return [IRBZ_points[i] for i in range(len(IRBZ_points)) if in_reduced_section[i]]

    def _get_orbit_labels(self, orbit_cosines_orig, key_points_inds_orbits, atol):
        orbit_cosines_copy = orbit_cosines_orig.copy()
        orbit_labels_unsorted = [(len(key_points_inds_orbits) - 1, 26)]
        orbit_inds_remaining = range(len(key_points_inds_orbits) - 1)
        pop_orbits = []
        pop_labels = []
        for i, orb_cos in enumerate(orbit_cosines_copy):
            if np.isclose(orb_cos[0][1], 1.0, atol=atol):
                orbit_labels_unsorted.append((i, orb_cos[0][0]))
                pop_orbits.append(i)
                pop_labels.append(orb_cos[0][0])
        orbit_cosines_copy = self._reduce_cosines_array(orbit_cosines_copy, pop_orbits, pop_labels)
        orbit_inds_remaining = [i for i in orbit_inds_remaining if i not in pop_orbits]
        while len(orbit_labels_unsorted) < len(orbit_cosines_orig) + 1:
            pop_orbits = []
            pop_labels = []
            max_cosine_value = max((orb_cos[0][1] for orb_cos in orbit_cosines_copy))
            max_cosine_value_inds = [j for j in range(len(orbit_cosines_copy)) if orbit_cosines_copy[j][0][1] == max_cosine_value]
            max_cosine_label_inds = self._get_max_cosine_labels([orbit_cosines_copy[j] for j in max_cosine_value_inds], key_points_inds_orbits, atol)
            for j, label_ind in enumerate(max_cosine_label_inds):
                orbit_labels_unsorted.append((orbit_inds_remaining[max_cosine_value_inds[j]], label_ind))
                pop_orbits.append(max_cosine_value_inds[j])
                pop_labels.append(label_ind)
            orbit_cosines_copy = self._reduce_cosines_array(orbit_cosines_copy, pop_orbits, pop_labels)
            orbit_inds_remaining = [orbit_inds_remaining[j] for j in range(len(orbit_inds_remaining)) if j not in pop_orbits]
        orbit_labels = np.zeros(len(key_points_inds_orbits))
        for tup in orbit_labels_unsorted:
            orbit_labels[tup[0]] = tup[1]
        return orbit_labels

    @staticmethod
    def _reduce_cosines_array(orbit_cosines, pop_orbits, pop_labels):
        return [[orb_cos[i] for i in range(len(orb_cos)) if orb_cos[i][0] not in pop_labels] for j, orb_cos in enumerate(orbit_cosines) if j not in pop_orbits]

    def _get_max_cosine_labels(self, max_cosine_orbits_orig, key_points_inds_orbits, atol):
        max_cosine_orbits_copy = max_cosine_orbits_orig.copy()
        max_cosine_label_inds = np.zeros(len(max_cosine_orbits_copy))
        initial_max_cosine_label_inds = [max_cos_orb[0][0] for max_cos_orb in max_cosine_orbits_copy]
        _uniq_vals, inds, counts = np.unique(initial_max_cosine_label_inds, return_index=True, return_counts=True)
        grouped_inds = [[i for i in range(len(initial_max_cosine_label_inds)) if max_cosine_orbits_copy[i][0][0] == max_cosine_orbits_copy[ind][0][0]] for ind in inds]
        pop_orbits = []
        pop_labels = []
        unassigned_orbits = []
        for idx, ind in enumerate(inds):
            if counts[idx] == 1:
                max_cosine_label_inds[ind] = initial_max_cosine_label_inds[ind]
                pop_orbits.append(ind)
                pop_labels.append(initial_max_cosine_label_inds[ind])
            else:
                next_choices = []
                for grouped_ind in grouped_inds[idx]:
                    j = 1
                    while True:
                        if max_cosine_orbits_copy[grouped_ind][j][0] not in initial_max_cosine_label_inds:
                            next_choices.append(max_cosine_orbits_copy[grouped_ind][j][1])
                            break
                        j += 1
                worst_next_choice = next_choices.index(min(next_choices))
                for grouped_ind in grouped_inds[idx]:
                    if grouped_ind != worst_next_choice:
                        unassigned_orbits.append(grouped_ind)
                max_cosine_label_inds[grouped_inds[idx][worst_next_choice]] = initial_max_cosine_label_inds[grouped_inds[idx][worst_next_choice]]
                pop_orbits.append(grouped_inds[idx][worst_next_choice])
                pop_labels.append(initial_max_cosine_label_inds[grouped_inds[idx][worst_next_choice]])
        if unassigned_orbits:
            max_cosine_orbits_copy = self._reduce_cosines_array(max_cosine_orbits_copy, pop_orbits, pop_labels)
            unassigned_orbits_labels = self._get_orbit_labels(max_cosine_orbits_copy, key_points_inds_orbits, atol)
            for idx, unassigned_orbit in enumerate(unassigned_orbits):
                max_cosine_label_inds[unassigned_orbit] = unassigned_orbits_labels[idx]
        return max_cosine_label_inds

    @staticmethod
    def label_points(index):
        """Axes used in generating labels for Latimer-Munro convention."""
        points = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1], [1, 2, 0], [1, 0, 2], [1, 2, 2], [2, 1, 0], [0, 1, 2], [2, 1, 2], [2, 0, 1], [0, 2, 1], [2, 2, 1], [1, 1, 2], [1, 2, 1], [2, 1, 1], [3, 3, 2], [3, 2, 3], [2, 3, 3], [2, 2, 2], [3, 2, 2], [2, 3, 2], [1e-10, 1e-10, 1e-10]]
        return points[index]

    @staticmethod
    def label_symbol(index):
        """Letters used in generating labels for the Latimer-Munro convention."""
        symbols = 'a b c d e f g h i j k l m n o p q r s t u v w x y z Γ'.split()
        return symbols[index]