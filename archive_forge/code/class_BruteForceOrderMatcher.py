from __future__ import annotations
import abc
import copy
import itertools
import logging
import math
import re
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import requires
from monty.json import MSONable
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from pymatgen.core.structure import Molecule
class BruteForceOrderMatcher(KabschMatcher):
    """Finding the best match between molecules by selecting molecule order
    with the smallest RMSD from all the possible order combinations.

    Notes:
        When aligning molecules, the atoms of the two molecules **must** have same number
        of atoms from the same species.
    """

    def match(self, mol: Molecule, ignore_warning: bool=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Similar as `KabschMatcher.match` but this method also finds the order of
        atoms which belongs to the best match.

        A `ValueError` will be raised when the total number of possible combinations
        become unfeasible (more than a million combination).

        Args:
            mol: a `Molecule` object what will be matched with the target one.
            ignore_warning: ignoring error when the number of combination is too large

        Returns:
            inds: The indices of atoms
            U: 3x3 rotation matrix
            V: Translation vector
            rmsd: Root mean squared deviation between P and Q
        """
        target_mol = self.target
        if sorted(mol.atomic_numbers) != sorted(target_mol.atomic_numbers):
            raise ValueError("The number of the same species aren't matching!")
        _, count = np.unique(mol.atomic_numbers, return_counts=True)
        total_permutations = 1
        for c in count:
            total_permutations *= math.factorial(c)
        if not ignore_warning and total_permutations > 1000000:
            raise ValueError(f'The number of all possible permutations ({total_permutations}) is not feasible to run this method!')
        p_coord, q_coord = (mol.cart_coords, target_mol.cart_coords)
        p_atoms, q_atoms = (np.array(mol.atomic_numbers), np.array(target_mol.atomic_numbers))
        p_trans, q_trans = (p_coord.mean(axis=0), q_coord.mean(axis=0))
        p_centroid, q_centroid = (p_coord - p_trans, q_coord - q_trans)
        q_inds = np.argsort(q_atoms)
        q_centroid = q_centroid[q_inds]
        rmsd = np.inf
        for p_inds_test in self.permutations(p_atoms):
            p_centroid_test = p_centroid[p_inds_test]
            U_test = self.kabsch(p_centroid_test, q_centroid)
            p_centroid_prime_test = np.dot(p_centroid_test, U_test)
            rmsd_test = np.sqrt(np.mean(np.square(p_centroid_prime_test - q_centroid)))
            if rmsd_test < rmsd:
                p_inds, U, rmsd = (p_inds_test, U_test, rmsd_test)
        V = q_trans - np.dot(p_trans, U)
        indices = p_inds[np.argsort(q_inds)]
        return (indices, U, V, rmsd)

    def fit(self, p: Molecule, ignore_warning=False):
        """Order, rotate and transform `p` molecule according to the best match.

        A `ValueError` will be raised when the total number of possible combinations
        become unfeasible (more than a million combinations).

        Args:
            p: a `Molecule` object what will be matched with the target one.
            ignore_warning: ignoring error when the number of combination is too large

        Returns:
            p_prime: Rotated and translated of the `p` `Molecule` object
            rmsd: Root-mean-square-deviation between `p_prime` and the `target`
        """
        inds, U, V, rmsd = self.match(p, ignore_warning=ignore_warning)
        p_prime = Molecule.from_sites([p[idx] for idx in inds])
        for site in p_prime:
            site.coords = np.dot(site.coords, U) + V
        return (p_prime, rmsd)

    @staticmethod
    def permutations(atoms):
        """Generates all the possible permutations of atom order. To achieve better
        performance all the cases where the atoms are different has been ignored.
        """
        element_iterators = [itertools.permutations(np.where(atoms == element)[0]) for element in np.unique(atoms)]
        for inds in itertools.product(*element_iterators):
            yield np.array(list(itertools.chain(*inds)))