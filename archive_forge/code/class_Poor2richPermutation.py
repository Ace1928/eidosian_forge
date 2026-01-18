import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
class Poor2richPermutation(_NeighborhoodPermutation):
    """The poor to rich (Poor2rich) permutation operator described in
    S. Lysgaard et al., Top. Catal., 2014, 57 (1-4), pp 33-39

    Permutes two atoms from regions short of the same elements, to
    regions rich in the same elements.
    (Inverse of Rich2poorPermutation)

    Parameters:

    elements: Which elements to take into account in this permutation

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, elements=[], num_muts=1, rng=np.random):
        _NeighborhoodPermutation.__init__(self, num_muts=num_muts, rng=rng)
        self.descriptor = 'Poor2richPermutation'
        self.elements = elements

    def get_new_individual(self, parents):
        f = parents[0].copy()
        diffatoms = len(set(f.numbers))
        assert diffatoms > 1, 'Permutations with one atomic type is not valid'
        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]
        for _ in range(self.num_muts):
            Poor2richPermutation.mutate(f, self.elements, rng=self.rng)
        for atom in f:
            indi.append(atom)
        return (self.finalize_individual(indi), self.descriptor + ':Parent {0}'.format(f.info['confid']))

    @classmethod
    def mutate(cls, atoms, elements, rng=np.random):
        _NP = _NeighborhoodPermutation
        ac = atoms.copy()
        del ac[[atom.index for atom in ac if atom.symbol not in elements]]
        permuts = _NP.get_possible_poor2rich_permutations(ac)
        swap = list(rng.choice(permuts))
        atoms.symbols[swap] = atoms.symbols[swap[::-1]]