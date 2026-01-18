from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
class RandomElementMutation(SlabOperator):

    def __init__(self, element_pools, verbose=False, num_muts=1, allowed_compositions=None, distribution_correction_function=None, rng=np.random):
        SlabOperator.__init__(self, verbose, num_muts, allowed_compositions, distribution_correction_function, element_pools=element_pools, rng=rng)
        self.descriptor = 'RandomElementMutation'

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.initialize_individual(f, self.operate(f))
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]
        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi), self.descriptor + parent_message)

    def operate(self, atoms):
        poss_muts = self.get_all_element_mutations(atoms)
        chosen = self.rng.randint(len(poss_muts))
        replace_element(atoms, *poss_muts[chosen])
        self.dcf(atoms)
        return atoms