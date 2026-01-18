import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
class _NeighborhoodPermutation(Mutation):
    """Helper class that holds common functions to all permutations
    that look at the neighborhoods of each atoms."""

    @classmethod
    def get_possible_poor2rich_permutations(cls, atoms, inverse=False, recurs=0, distance_matrix=None):
        dm = distance_matrix
        if dm is None:
            dm = get_distance_matrix(atoms)
        nndist = get_nndist(atoms, dm) + 0.2
        same_neighbors = {}

        def f(x):
            return x[1]
        for i, atom in enumerate(atoms):
            same_neighbors[i] = 0
            neighbors = [j for j in range(len(dm[i])) if dm[i][j] < nndist]
            for n in neighbors:
                if atoms[n].symbol == atom.symbol:
                    same_neighbors[i] += 1
        sorted_same = sorted(same_neighbors.items(), key=f)
        if inverse:
            sorted_same.reverse()
        poor_indices = [j[0] for j in sorted_same if abs(j[1] - sorted_same[0][1]) <= recurs]
        rich_indices = [j[0] for j in sorted_same if abs(j[1] - sorted_same[-1][1]) <= recurs]
        permuts = Mutation.get_list_of_possible_permutations(atoms, poor_indices, rich_indices)
        if len(permuts) == 0:
            _NP = _NeighborhoodPermutation
            return _NP.get_possible_poor2rich_permutations(atoms, inverse, recurs + 1, dm)
        return permuts