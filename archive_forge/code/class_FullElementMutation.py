import numpy as np
from ase.data import atomic_numbers
from ase.ga.offspring_creator import OffspringCreator
class FullElementMutation(OffspringCreator):
    """Mutation that exchanges an all elements of a certain type with another
    randomly chosen element from the supplied pool of elements. Any constraints
    on the mutation are inhereted from the original candidate.

    Parameters:

    element_pool: List of elements in the phase space. The elements can be
        grouped if the individual consist of different types of elements.
        The list should then be a list of lists e.g. [[list1], [list2]]

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, element_pool, verbose=False, num_muts=1, rng=np.random):
        OffspringCreator.__init__(self, verbose, num_muts=num_muts, rng=rng)
        self.descriptor = 'FullElementMutation'
        if not isinstance(element_pool[0], (list, np.ndarray)):
            self.element_pools = [element_pool]
        else:
            self.element_pools = element_pool

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.initialize_individual(f)
        indi.info['data']['parents'] = [f.info['confid']]
        old_element = self.rng.choice([a.symbol for a in f])
        for i in range(len(self.element_pools)):
            if old_element in self.element_pools[i]:
                lm = i
        not_val = True
        while not_val:
            new_element = self.rng.choice(self.element_pools[lm])
            not_val = new_element == old_element
        for a in f:
            if a.symbol == old_element:
                a.symbol = new_element
            indi.append(a)
        return (self.finalize_individual(indi), self.descriptor + ': Parent {0}'.format(f.info['confid']))