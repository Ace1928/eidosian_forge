from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class ForceConstantMatrix:
    """
    This class describes the NxNx3x3 force constant matrix defined by a
    structure, point operations of the structure's atomic sites, and the
    shared symmetry operations between pairs of atomic sites.
    """

    def __init__(self, structure: Structure, fcm, pointops, sharedops, tol: float=0.001):
        """
        Create an ForceConstantMatrix object.

        Args:
            input_matrix (NxNx3x3 array-like): the NxNx3x3 array-like
                representing the force constant matrix
        """
        self.structure = structure
        self.fcm = fcm
        self.pointops = pointops
        self.sharedops = sharedops
        self.FCM_operations = None

    def get_FCM_operations(self, eigtol=1e-05, opstol=1e-05):
        """
        Returns the symmetry operations which maps the tensors
        belonging to equivalent sites onto each other in the form
        [site index 1a, site index 1b, site index 2a, site index 2b,
        [Symmops mapping from site index 1a, 1b to site index 2a, 2b]].

        Args:
            eigtol (float): tolerance for determining if two sites are
            related by symmetry
            opstol (float): tolerance for determining if a symmetry
            operation relates two sites

        Returns:
            list of symmetry operations mapping equivalent sites and
            the indexes of those sites.
        """
        struct = self.structure
        ops = SpacegroupAnalyzer(struct).get_symmetry_operations(cartesian=True)
        uniq_point_ops = list(ops)
        for ops in self.pointops:
            for op in ops:
                if op not in uniq_point_ops:
                    uniq_point_ops.append(op)
        passed = []
        relations = []
        for atom1 in range(len(self.fcm)):
            for atom2 in range(atom1, len(self.fcm)):
                unique = 1
                eig1, _vecs1 = np.linalg.eig(self.fcm[atom1][atom2])
                index = np.argsort(eig1)
                new_eig = np.real([eig1[index[0]], eig1[index[1]], eig1[index[2]]])
                for p in passed:
                    if np.allclose(new_eig, p[2], atol=eigtol):
                        relations.append([atom1, atom2, p[0], p[1]])
                        unique = 0
                        break
                if unique == 1:
                    relations.append([atom1, atom2, atom2, atom1])
                    passed.append([atom1, atom2, np.real(new_eig)])
        FCM_operations = []
        for entry, r in enumerate(relations):
            FCM_operations.append(r)
            FCM_operations[entry].append([])
            good = 0
            for op in uniq_point_ops:
                new = op.transform_tensor(self.fcm[r[2]][r[3]])
                if np.allclose(new, self.fcm[r[0]][r[1]], atol=opstol):
                    FCM_operations[entry][4].append(op)
                    good = 1
            if r[0] == r[3] and r[1] == r[2]:
                good = 1
            if r[0] == r[2] and r[1] == r[3]:
                good = 1
            if good == 0:
                FCM_operations[entry] = [r[0], r[1], r[3], r[2]]
                FCM_operations[entry].append([])
                for op in uniq_point_ops:
                    new = op.transform_tensor(self.fcm[r[2]][r[3]])
                    if np.allclose(new.T, self.fcm[r[0]][r[1]], atol=opstol):
                        FCM_operations[entry][4].append(op)
        self.FCM_operations = FCM_operations
        return FCM_operations

    def get_unstable_FCM(self, max_force=1):
        """
        Generate an unsymmetrized force constant matrix.

        Args:
            max_charge (float): maximum born effective charge value

        Returns:
            numpy array representing the force constant matrix
        """
        struct = self.structure
        operations = self.FCM_operations
        n_sites = len(struct)
        D = 1 / max_force * 2 * np.ones([n_sites * 3, n_sites * 3])
        for op in operations:
            same = 0
            transpose = 0
            if op[0] == op[1] and op[0] == op[2] and (op[0] == op[3]):
                same = 1
            if op[0] == op[3] and op[1] == op[2]:
                transpose = 1
            if transpose == 0 and same == 0:
                D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = np.zeros([3, 3])
                D[3 * op[1]:3 * op[1] + 3, 3 * op[0]:3 * op[0] + 3] = np.zeros([3, 3])
                for symop in op[4]:
                    temp_fcm = D[3 * op[2]:3 * op[2] + 3, 3 * op[3]:3 * op[3] + 3]
                    temp_fcm = symop.transform_tensor(temp_fcm)
                    D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] += temp_fcm
                if len(op[4]) != 0:
                    D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] / len(op[4])
                D[3 * op[1]:3 * op[1] + 3, 3 * op[0]:3 * op[0] + 3] = D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3].T
                continue
            temp_tensor = Tensor(np.random.rand(3, 3) - 0.5) * max_force
            temp_tensor_sum = sum((temp_tensor.transform(symm_op) for symm_op in self.sharedops[op[0]][op[1]]))
            temp_tensor_sum = temp_tensor_sum / len(self.sharedops[op[0]][op[1]])
            if op[0] != op[1]:
                for pair in range(len(op[4])):
                    temp_tensor2 = temp_tensor_sum.T
                    temp_tensor2 = op[4][pair].transform_tensor(temp_tensor2)
                    temp_tensor_sum = (temp_tensor_sum + temp_tensor2) / 2
            else:
                temp_tensor_sum = (temp_tensor_sum + temp_tensor_sum.T) / 2
            D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = temp_tensor_sum
            D[3 * op[1]:3 * op[1] + 3, 3 * op[0]:3 * op[0] + 3] = temp_tensor_sum.T
        return D

    def get_symmetrized_FCM(self, unsymmetrized_fcm, max_force=1):
        """
        Generate a symmetrized force constant matrix from an unsymmetrized matrix.

        Args:
            unsymmetrized_fcm (numpy array): unsymmetrized force constant matrix
            max_charge (float): maximum born effective charge value

        Returns:
            3Nx3N numpy array representing the force constant matrix
        """
        operations = self.FCM_operations
        for op in operations:
            same = 0
            transpose = 0
            if op[0] == op[1] and op[0] == operations[2] and (op[0] == op[3]):
                same = 1
            if op[0] == op[3] and op[1] == op[2]:
                transpose = 1
            if transpose == 0 and same == 0:
                unsymmetrized_fcm[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = np.zeros([3, 3])
                for symop in op[4]:
                    tempfcm = unsymmetrized_fcm[3 * op[2]:3 * op[2] + 3, 3 * op[3]:3 * op[3] + 3]
                    tempfcm = symop.transform_tensor(tempfcm)
                    unsymmetrized_fcm[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] += tempfcm
                if len(op[4]) != 0:
                    unsymmetrized_fcm[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = unsymmetrized_fcm[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] / len(op[4])
                unsymmetrized_fcm[3 * op[1]:3 * op[1] + 3, 3 * op[0]:3 * op[0] + 3] = unsymmetrized_fcm[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3].T
                continue
            temp_tensor = Tensor(unsymmetrized_fcm[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3])
            temp_tensor_sum = sum((temp_tensor.transform(symm_op) for symm_op in self.sharedops[op[0]][op[1]]))
            if len(self.sharedops[op[0]][op[1]]) != 0:
                temp_tensor_sum = temp_tensor_sum / len(self.sharedops[op[0]][op[1]])
            if op[0] != op[1]:
                for pair in range(len(op[4])):
                    temp_tensor2 = temp_tensor_sum.T
                    temp_tensor2 = op[4][pair].transform_tensor(temp_tensor2)
                    temp_tensor_sum = (temp_tensor_sum + temp_tensor2) / 2
            else:
                temp_tensor_sum = (temp_tensor_sum + temp_tensor_sum.T) / 2
            unsymmetrized_fcm[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = temp_tensor_sum
            unsymmetrized_fcm[3 * op[1]:3 * op[1] + 3, 3 * op[0]:3 * op[0] + 3] = temp_tensor_sum.T
        return unsymmetrized_fcm

    def get_stable_FCM(self, fcm, fcmasum=10):
        """
        Generate a symmetrized force constant matrix that obeys the objects symmetry
        constraints, has no unstable modes and also obeys the acoustic sum rule through an
        iterative procedure.

        Args:
            fcm (numpy array): unsymmetrized force constant matrix
            fcmasum (int): number of iterations to attempt to obey the acoustic sum
                rule

        Returns:
            3Nx3N numpy array representing the force constant matrix
        """
        check = 0
        count = 0
        while check == 0:
            if count > 20:
                check = 1
                break
            eigs, vecs = np.linalg.eig(fcm)
            max_eig = np.max(-1 * eigs)
            eig_sort = np.argsort(np.abs(eigs))
            for idx in range(3, len(eigs)):
                if eigs[eig_sort[idx]] > 1e-06:
                    eigs[eig_sort[idx]] = -1 * max_eig * np.random.rand()
            diag = np.real(np.eye(len(fcm)) * eigs)
            fcm = np.real(np.matmul(np.matmul(vecs, diag), vecs.T))
            fcm = self.get_symmetrized_FCM(fcm)
            fcm = self.get_asum_FCM(fcm)
            eigs, vecs = np.linalg.eig(fcm)
            unstable_modes = 0
            eig_sort = np.argsort(np.abs(eigs))
            for idx in range(3, len(eigs)):
                if eigs[eig_sort[idx]] > 1e-06:
                    unstable_modes = 1
            if unstable_modes == 1:
                count = count + 1
                continue
            check = 1
        return fcm

    def get_asum_FCM(self, fcm: np.ndarray, numiter: int=15):
        """
        Generate a symmetrized force constant matrix that obeys the objects symmetry
        constraints and obeys the acoustic sum rule through an iterative procedure.

        Args:
            fcm (numpy array): 3Nx3N unsymmetrized force constant matrix
            numiter (int): number of iterations to attempt to obey the acoustic sum
                rule

        Returns:
            numpy array representing the force constant matrix
        """
        operations = self.FCM_operations
        if operations is None:
            raise RuntimeError('No symmetry operations found. Run get_FCM_operations first.')
        n_sites = len(self.structure)
        D = np.ones([n_sites * 3, n_sites * 3])
        for _ in range(numiter):
            X = np.real(fcm)
            pastrow = 0
            total = np.zeros([3, 3])
            for col in range(n_sites):
                total = total + X[0:3, col * 3:col * 3 + 3]
            total = total / n_sites
            for op in operations:
                same = 0
                transpose = 0
                if op[0] == op[1] and op[0] == op[2] and (op[0] == op[3]):
                    same = 1
                if op[0] == op[3] and op[1] == op[2]:
                    transpose = 1
                if transpose == 0 and same == 0:
                    D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = np.zeros([3, 3])
                    for symop in op[4]:
                        tempfcm = D[3 * op[2]:3 * op[2] + 3, 3 * op[3]:3 * op[3] + 3]
                        tempfcm = symop.transform_tensor(tempfcm)
                        D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] += tempfcm
                    if len(op[4]) != 0:
                        D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] / len(op[4])
                    D[3 * op[1]:3 * op[1] + 3, 3 * op[0]:3 * op[0] + 3] = D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3].T
                    continue
                curr_row = op[0]
                if curr_row != pastrow:
                    total = np.zeros([3, 3])
                    for col in range(n_sites):
                        total = total + X[curr_row * 3:curr_row * 3 + 3, col * 3:col * 3 + 3]
                    for col in range(curr_row):
                        total = total - D[curr_row * 3:curr_row * 3 + 3, col * 3:col * 3 + 3]
                    total = total / (n_sites - curr_row)
                pastrow = curr_row
                temp_tensor = Tensor(total)
                temp_tensor_sum = sum((temp_tensor.transform(symm_op) for symm_op in self.sharedops[op[0]][op[1]]))
                if len(self.sharedops[op[0]][op[1]]) != 0:
                    temp_tensor_sum = temp_tensor_sum / len(self.sharedops[op[0]][op[1]])
                if op[0] != op[1]:
                    for pair in range(len(op[4])):
                        temp_tensor2 = temp_tensor_sum.T
                        temp_tensor2 = op[4][pair].transform_tensor(temp_tensor2)
                        temp_tensor_sum = (temp_tensor_sum + temp_tensor2) / 2
                else:
                    temp_tensor_sum = (temp_tensor_sum + temp_tensor_sum.T) / 2
                D[3 * op[0]:3 * op[0] + 3, 3 * op[1]:3 * op[1] + 3] = temp_tensor_sum
                D[3 * op[1]:3 * op[1] + 3, 3 * op[0]:3 * op[0] + 3] = temp_tensor_sum.T
            fcm = fcm - D
        return fcm

    @requires(Phonopy, 'phonopy not installed!')
    def get_rand_FCM(self, asum=15, force=10):
        """
        Generate a symmetrized force constant matrix from an unsymmetrized matrix
        that has no unstable modes and also obeys the acoustic sum rule through an
        iterative procedure.

        Args:
            force (float): maximum force constant
            asum (int): number of iterations to attempt to obey the acoustic sum
                rule

        Returns:
            NxNx3x3 np.array representing the force constant matrix
        """
        from pymatgen.io.phonopy import get_phonopy_structure
        n_sites = len(self.structure)
        structure = get_phonopy_structure(self.structure)
        pn_struct = Phonopy(structure, np.eye(3), np.eye(3))
        dyn = self.get_unstable_FCM(force)
        dyn = self.get_stable_FCM(dyn)
        dyn = np.reshape(dyn, (n_sites, 3, n_sites, 3)).swapaxes(1, 2)
        dyn_mass = np.zeros([len(self.structure), len(self.structure), 3, 3])
        masses = []
        for idx in range(n_sites):
            masses.append(self.structure[idx].specie.atomic_mass)
        dyn_mass = np.zeros([n_sites, n_sites, 3, 3])
        for m in range(n_sites):
            for n in range(n_sites):
                dyn_mass[m][n] = dyn[m][n] * np.sqrt(masses[m]) * np.sqrt(masses[n])
        supercell = pn_struct.get_supercell()
        primitive = pn_struct.get_primitive()
        converter = dyntofc.DynmatToForceConstants(primitive, supercell)
        dyn = np.reshape(np.swapaxes(dyn_mass, 1, 2), (n_sites * 3, n_sites * 3))
        converter.set_dynamical_matrices(dynmat=[dyn])
        converter.run()
        return converter.get_force_constants()