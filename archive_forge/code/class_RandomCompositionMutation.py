from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
class RandomCompositionMutation(SlabOperator):
    """Change the current composition to another of the allowed compositions.
    The allowed compositions should be input in the same order as the element pools,
    for example:
    element_pools = [['Au', 'Cu'], ['In', 'Bi']]
    allowed_compositions = [(6, 2), (5, 3)]
    means that there can be 5 or 6 Au and Cu, and 2 or 3 In and Bi.
    """

    def __init__(self, verbose=False, num_muts=1, element_pools=None, allowed_compositions=None, distribution_correction_function=None, rng=np.random):
        SlabOperator.__init__(self, verbose, num_muts, allowed_compositions, distribution_correction_function, element_pools=element_pools, rng=rng)
        self.descriptor = 'RandomCompositionMutation'

    def get_new_individual(self, parents):
        f = parents[0]
        parent_message = ': Parent {0}'.format(f.info['confid'])
        if self.allowed_compositions is None:
            if len(set(f.get_chemical_symbols())) == 1:
                if self.element_pools is None:
                    return (None, self.descriptor + parent_message)
        indi = self.initialize_individual(f, self.operate(f))
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]
        return (self.finalize_individual(indi), self.descriptor + parent_message)

    def operate(self, atoms):
        allowed_comps = self.allowed_compositions
        if allowed_comps is None:
            n_elems = len(set(atoms.get_chemical_symbols()))
            n_atoms = len(atoms)
            allowed_comps = [c for c in permutations(range(1, n_atoms), n_elems) if sum(c) == n_atoms]
        syms = atoms.get_chemical_symbols()
        unique_syms, _, comp = get_ordered_composition(syms, self.element_pools)
        for i, allowed in enumerate(allowed_comps):
            if comp == tuple(allowed):
                allowed_comps = np.delete(allowed_comps, i, axis=0)
                break
        chosen = self.rng.randint(len(allowed_comps))
        comp_diff = self.get_composition_diff(comp, allowed_comps[chosen])
        to_add, to_rem = get_add_remove_lists(**dict(zip(unique_syms, comp_diff)))
        syms = atoms.get_chemical_symbols()
        for add, rem in zip(to_add, to_rem):
            tbc = [i for i in range(len(syms)) if syms[i] == rem]
            ai = self.rng.choice(tbc)
            syms[ai] = add
        atoms.set_chemical_symbols(syms)
        self.dcf(atoms)
        return atoms