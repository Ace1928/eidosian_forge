import numpy as np
from math import cos, sin, pi
from ase.calculators.lammpslib import convert_cell
from ase.ga.utilities import (atoms_too_close,
from ase.ga.offspring_creator import OffspringCreator, CombinationMutation
from ase import Atoms
class PermutationMutation(OffspringCreator):
    """Mutation that permutes a percentage of the atom types in the cluster.

    Parameters:

    n_top: Number of atoms optimized by the GA.

    probability: The probability with which an atom is permuted.

    test_dist_to_slab: whether to also make sure that the distances
        between the atoms and the slab satisfy the blmin.

    use_tags: if True, the atomic tags will be used to preserve
        molecular identity. Permutations will then happen
        at the molecular level, i.e. swapping the center-of-
        positions of two moieties while preserving their
        internal geometries.

    blmin: Dictionary defining the minimum distance between atoms
        after the permutation. If equal to None (the default),
        no such check is performed.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, n_top, probability=0.33, test_dist_to_slab=True, use_tags=False, blmin=None, rng=np.random, verbose=False):
        OffspringCreator.__init__(self, verbose, rng=rng)
        self.n_top = n_top
        self.probability = probability
        self.test_dist_to_slab = test_dist_to_slab
        self.use_tags = use_tags
        self.blmin = blmin
        self.descriptor = 'PermutationMutation'
        self.min_inputs = 1

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.mutate(f)
        if indi is None:
            return (indi, 'mutation: permutation')
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]
        return (self.finalize_individual(indi), 'mutation: permutation')

    def mutate(self, atoms):
        """Does the actual mutation."""
        N = len(atoms) if self.n_top is None else self.n_top
        slab = atoms[:len(atoms) - N]
        atoms = atoms[-N:]
        if self.use_tags:
            gather_atoms_by_tag(atoms)
        tags = atoms.get_tags() if self.use_tags else np.arange(N)
        pos_ref = atoms.get_positions()
        num = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        symbols = atoms.get_chemical_symbols()
        unique_tags = np.unique(tags)
        n = len(unique_tags)
        swaps = int(np.ceil(n * self.probability / 2.0))
        sym = []
        for tag in unique_tags:
            indices = np.where(tags == tag)[0]
            s = ''.join([symbols[j] for j in indices])
            sym.append(s)
        assert len(np.unique(sym)) > 1, 'Permutations with one atom (or molecule) type is not valid'
        count = 0
        maxcount = 1000
        too_close = True
        while too_close and count < maxcount:
            count += 1
            pos = pos_ref.copy()
            for _ in range(swaps):
                i = j = 0
                while sym[i] == sym[j]:
                    i = self.rng.randint(0, high=n)
                    j = self.rng.randint(0, high=n)
                ind1 = np.where(tags == i)
                ind2 = np.where(tags == j)
                cop1 = np.mean(pos[ind1], axis=0)
                cop2 = np.mean(pos[ind2], axis=0)
                pos[ind1] += cop2 - cop1
                pos[ind2] += cop1 - cop2
            top = Atoms(num, positions=pos, cell=cell, pbc=pbc, tags=tags)
            if self.blmin is None:
                too_close = False
            else:
                too_close = atoms_too_close(top, self.blmin, use_tags=self.use_tags)
                if not too_close and self.test_dist_to_slab:
                    too_close = atoms_too_close_two_sets(top, slab, self.blmin)
        if count == maxcount:
            return None
        mutant = slab + top
        return mutant