import numpy as np
from operator import itemgetter
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.utilities import get_distance_matrix, get_nndist
from ase import Atoms
class COM2surfPermutation(Mutation):
    """The Center Of Mass to surface (COM2surf) permutation operator
    described in
    S. Lysgaard et al., Top. Catal., 2014, 57 (1-4), pp 33-39

    Parameters:

    elements: which elements should be included in this permutation,
        for example: include all metals and exclude all adsorbates

    min_ratio: minimum ratio of each element in the core or surface region.
        If elements=[a, b] then ratio of a is Na / (Na + Nb) (N: Number of).
        If less than minimum ratio is present in the core, the region defining
        the core will be extended until the minimum ratio is met, and vice
        versa for the surface region. It has the potential reach the
        recursive limit if an element has a smaller total ratio in the
        complete particle. In that case remember to decrease this min_ratio.

    num_muts: the number of times to perform this operation.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, elements=None, min_ratio=0.25, num_muts=1, rng=np.random):
        Mutation.__init__(self, num_muts=num_muts, rng=rng)
        self.descriptor = 'COM2surfPermutation'
        self.min_ratio = min_ratio
        self.elements = elements

    def get_new_individual(self, parents):
        f = parents[0].copy()
        diffatoms = len(set(f.numbers))
        assert diffatoms > 1, 'Permutations with one atomic type is not valid'
        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]
        for _ in range(self.num_muts):
            elems = self.elements
            COM2surfPermutation.mutate(f, elems, self.min_ratio, rng=self.rng)
        for atom in f:
            indi.append(atom)
        return (self.finalize_individual(indi), self.descriptor + ':Parent {0}'.format(f.info['confid']))

    @classmethod
    def mutate(cls, atoms, elements, min_ratio, rng=np.random):
        """Performs the COM2surf permutation."""
        ac = atoms.copy()
        if elements is not None:
            del ac[[a.index for a in ac if a.symbol not in elements]]
        syms = ac.get_chemical_symbols()
        for el in set(syms):
            assert syms.count(el) / float(len(syms)) > min_ratio
        atomic_conf = Mutation.get_atomic_configuration(atoms, elements=elements)
        core = COM2surfPermutation.get_core_indices(atoms, atomic_conf, min_ratio)
        shell = COM2surfPermutation.get_shell_indices(atoms, atomic_conf, min_ratio)
        permuts = Mutation.get_list_of_possible_permutations(atoms, core, shell)
        chosen = rng.randint(len(permuts))
        swap = list(permuts[chosen])
        atoms.symbols[swap] = atoms.symbols[swap[::-1]]

    @classmethod
    def get_core_indices(cls, atoms, atomic_conf, min_ratio, recurs=0):
        """Recursive function that returns the indices in the core subject to
        the min_ratio constraint. The indices are found from the supplied
        atomic configuration."""
        elements = list(set([atoms[i].symbol for subl in atomic_conf for i in subl]))
        core = [i for subl in atomic_conf[:1 + recurs] for i in subl]
        while len(core) < 1:
            recurs += 1
            core = [i for subl in atomic_conf[:1 + recurs] for i in subl]
        for elem in elements:
            ratio = len([i for i in core if atoms[i].symbol == elem]) / float(len(core))
            if ratio < min_ratio:
                return COM2surfPermutation.get_core_indices(atoms, atomic_conf, min_ratio, recurs + 1)
        return core

    @classmethod
    def get_shell_indices(cls, atoms, atomic_conf, min_ratio, recurs=0):
        """Recursive function that returns the indices in the surface
        subject to the min_ratio constraint. The indices are found from
        the supplied atomic configuration."""
        elements = list(set([atoms[i].symbol for subl in atomic_conf for i in subl]))
        shell = [i for subl in atomic_conf[-1 - recurs:] for i in subl]
        while len(shell) < 1:
            recurs += 1
            shell = [i for subl in atomic_conf[-1 - recurs:] for i in subl]
        for elem in elements:
            ratio = len([i for i in shell if atoms[i].symbol == elem]) / float(len(shell))
            if ratio < min_ratio:
                return COM2surfPermutation.get_shell_indices(atoms, atomic_conf, min_ratio, recurs + 1)
        return shell