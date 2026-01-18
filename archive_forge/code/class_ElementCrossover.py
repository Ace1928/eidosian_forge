import numpy as np
from ase.ga.offspring_creator import OffspringCreator
class ElementCrossover(OffspringCreator):
    """Base class for all operators where the elements of
    the atoms objects cross.

    """

    def __init__(self, element_pool, max_diff_elements, min_percentage_elements, verbose, rng=np.random):
        OffspringCreator.__init__(self, verbose, rng=rng)
        if not isinstance(element_pool[0], (list, np.ndarray)):
            self.element_pools = [element_pool]
        else:
            self.element_pools = element_pool
        if max_diff_elements is None:
            self.max_diff_elements = [None for _ in self.element_pools]
        elif isinstance(max_diff_elements, int):
            self.max_diff_elements = [max_diff_elements]
        else:
            self.max_diff_elements = max_diff_elements
        assert len(self.max_diff_elements) == len(self.element_pools)
        if min_percentage_elements is None:
            self.min_percentage_elements = [0 for _ in self.element_pools]
        elif isinstance(min_percentage_elements, (int, float)):
            self.min_percentage_elements = [min_percentage_elements]
        else:
            self.min_percentage_elements = min_percentage_elements
        assert len(self.min_percentage_elements) == len(self.element_pools)
        self.min_inputs = 2

    def get_new_individual(self, parents):
        raise NotImplementedError