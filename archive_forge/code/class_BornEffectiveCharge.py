from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from pymatgen.core.tensors import Tensor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
class BornEffectiveCharge:
    """This class describes the Nx3x3 born effective charge tensor."""

    def __init__(self, structure: Structure, bec, pointops, tol: float=0.001):
        """
        Create an BornEffectiveChargeTensor object defined by a
        structure, point operations of the structure's atomic sites.
        Note that the constructor uses __new__ rather than __init__
        according to the standard method of subclassing numpy ndarrays.

        Args:
            input_matrix (Nx3x3 array-like): the Nx3x3 array-like
                representing the born effective charge tensor
        """
        self.structure = structure
        self.bec = bec
        self.pointops = pointops
        self.BEC_operations = None
        if np.sum(self.bec) >= tol:
            warnings.warn('Input born effective charge tensor does not satisfy charge neutrality')

    def get_BEC_operations(self, eigtol=1e-05, opstol=0.001):
        """
        Returns the symmetry operations which maps the tensors
        belonging to equivalent sites onto each other in the form
        [site index 1, site index 2, [Symmops mapping from site
        index 1 to site index 2]].

        Args:
            eigtol (float): tolerance for determining if two sites are
            related by symmetry
            opstol (float): tolerance for determining if a symmetry
            operation relates two sites

        Returns:
            list of symmetry operations mapping equivalent sites and
            the indexes of those sites.
        """
        bec = self.bec
        struct = self.structure
        ops = SpacegroupAnalyzer(struct).get_symmetry_operations(cartesian=True)
        uniq_point_ops = list(ops)
        for ops in self.pointops:
            for op in ops:
                if op not in uniq_point_ops:
                    uniq_point_ops.append(op)
        passed = []
        relations = []
        for site, val in enumerate(bec):
            unique = 1
            eig1, _vecs1 = np.linalg.eig(val)
            index = np.argsort(eig1)
            new_eig = np.real([eig1[index[0]], eig1[index[1]], eig1[index[2]]])
            for index, p in enumerate(passed):
                if np.allclose(new_eig, p[1], atol=eigtol):
                    relations.append([site, index])
                    unique = 0
                    passed.append([site, p[0], new_eig])
                    break
            if unique == 1:
                relations.append([site, site])
                passed.append([site, new_eig])
        BEC_operations = []
        for atom, r in enumerate(relations):
            BEC_operations.append(r)
            BEC_operations[atom].append([])
            for op in uniq_point_ops:
                new = op.transform_tensor(self.bec[relations[atom][1]])
                if np.allclose(new, self.bec[r[0]], atol=opstol):
                    BEC_operations[atom][2].append(op)
        self.BEC_operations = BEC_operations
        return BEC_operations

    def get_rand_BEC(self, max_charge=1):
        """
        Generate a random born effective charge tensor which obeys a structure's
        symmetry and the acoustic sum rule.

        Args:
            max_charge (float): maximum born effective charge value

        Returns:
            np.array Born effective charge tensor
        """
        n_atoms = len(self.structure)
        BEC = np.zeros((n_atoms, 3, 3))
        for atom, ops in enumerate(self.BEC_operations):
            if ops[0] == ops[1]:
                temp_tensor = Tensor(np.random.rand(3, 3) - 0.5)
                temp_tensor = sum((temp_tensor.transform(symm_op) for symm_op in self.pointops[atom])) / len(self.pointops[atom])
                BEC[atom] = temp_tensor
            else:
                temp_fcm = np.zeros([3, 3])
                for op in ops[2]:
                    temp_fcm += op.transform_tensor(BEC[self.BEC_operations[atom][1]])
                BEC[ops[0]] = temp_fcm
                if len(ops[2]) != 0:
                    BEC[ops[0]] = BEC[ops[0]] / len(ops[2])
        disp_charge = np.einsum('ijk->jk', BEC) / n_atoms
        add = np.zeros([n_atoms, 3, 3])
        for atom, ops in enumerate(self.BEC_operations):
            if ops[0] == ops[1]:
                temp_tensor = Tensor(disp_charge)
                temp_tensor = sum((temp_tensor.transform(symm_op) for symm_op in self.pointops[atom])) / len(self.pointops[atom])
                add[ops[0]] = temp_tensor
            else:
                temp_tensor = np.zeros([3, 3])
                for op in ops[2]:
                    temp_tensor += op.transform_tensor(add[self.BEC_operations[atom][1]])
                add[ops[0]] = temp_tensor
                if len(ops) != 0:
                    add[ops[0]] = add[ops[0]] / len(ops[2])
        BEC = BEC - add
        return BEC * max_charge