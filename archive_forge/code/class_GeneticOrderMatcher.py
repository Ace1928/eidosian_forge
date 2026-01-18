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
class GeneticOrderMatcher(KabschMatcher):
    """This method was inspired by genetic algorithms and tries to match molecules
    based on their already matched fragments.

    It uses the fact that when two molecule is matching their sub-structures have to match as well.
    The main idea here is that in each iteration (generation) we can check the match of all possible
    fragments and ignore those which are not feasible.

    Although in the worst case this method has N! complexity (same as the brute force one),
    in practice it performs much faster because many of the combination can be eliminated
    during the fragment matching.

    Notes:
        This method very robust and returns with all the possible orders.

        There is a well known weakness/corner case: The case when there is
        a outlier with large deviation with a small index might be ignored.
        This happens due to the nature of the average function
        used to calculate the RMSD for the fragments.

        When aligning molecules, the atoms of the two molecules **must** have the
        same number of atoms from the same species.
    """

    def __init__(self, target: Molecule, threshold: float):
        """Constructor of the matcher object.

        Args:
            target: a `Molecule` object used as a target during the alignment
            threshold: value used to match fragments and prune configuration
        """
        super().__init__(target)
        self.threshold = threshold
        self.N = len(target)

    def match(self, p: Molecule):
        """Similar as `KabschMatcher.match` but this method also finds all of the
        possible atomic orders according to the `threshold`.

        Args:
            p: a `Molecule` object what will be matched with the target one.

        Returns:
            Array of the possible matches where the elements are:
                inds: The indices of atoms
                U: 3x3 rotation matrix
                V: Translation vector
                rmsd: Root mean squared deviation between P and Q
        """
        out = []
        for inds in self.permutations(p):
            p_prime = p.copy()
            p_prime._sites = [p_prime[idx] for idx in inds]
            U, V, rmsd = super().match(p_prime)
            out.append((inds, U, V, rmsd))
        return out

    def fit(self, p: Molecule):
        """Order, rotate and transform all of the matched `p` molecule
        according to the given `threshold`.

        Args:
            p: a `Molecule` object what will be matched with the target one.

        Returns:
            list[tuple[Molecule, float]]: possible matches where the elements are:
                p_prime: Rotated and translated of the `p` `Molecule` object
                rmsd: Root-mean-square-deviation between `p_prime` and the `target`
        """
        out: list[tuple[Molecule, float]] = []
        for inds in self.permutations(p):
            p_prime = p.copy()
            p_prime._sites = [p_prime[idx] for idx in inds]
            U, V, rmsd = super().match(p_prime)
            for site in p_prime:
                site.coords = np.dot(site.coords, U) + V
            out += [(p_prime, rmsd)]
        return out

    def permutations(self, p: Molecule):
        """Generates all of possible permutations of atom order according the threshold.

        Args:
            p: a `Molecule` object what will be matched with the target one.

        Returns:
            Array of index arrays
        """
        p_atoms, q_atoms = (p.atomic_numbers, self.target.atomic_numbers)
        p_coords, q_coords = (p.cart_coords, self.target.cart_coords)
        if sorted(p_atoms) != sorted(q_atoms):
            raise ValueError("The number of the same species aren't matching!")
        partial_matches = [[j] for j in range(self.N) if p_atoms[j] == q_atoms[0]]
        for idx in range(1, self.N):
            f_coords = q_coords[:idx + 1]
            f_atom = q_atoms[idx]
            f_trans = f_coords.mean(axis=0)
            f_centroid = f_coords - f_trans
            matches = []
            for indices in partial_matches:
                for jdx in range(self.N):
                    if jdx in indices:
                        continue
                    if p_atoms[jdx] != f_atom:
                        continue
                    inds = [*indices, jdx]
                    P = p_coords[inds]
                    p_trans = P.mean(axis=0)
                    p_centroid = P - p_trans
                    U = self.kabsch(p_centroid, f_centroid)
                    p_prime_centroid = np.dot(p_centroid, U)
                    rmsd = np.sqrt(np.mean(np.square(p_prime_centroid - f_centroid)))
                    if rmsd > self.threshold:
                        continue
                    logger.debug(f'match - rmsd={rmsd!r}, inds={inds!r}')
                    matches.append(inds)
            partial_matches = matches
            logger.info(f'number of atom in the fragment: {idx + 1}, number of possible matches: {len(matches)}')
        return matches