import numpy as np
from ase import Atoms
class OperationSelector:
    """Class used to randomly select a procreation operation
    from a list of operations.

    Parameters:

    probabilities: A list of probabilities with which the different
        mutations should be selected. The norm of this list
        does not need to be 1.

    oplist: The list of operations to select from.

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, probabilities, oplist, rng=np.random):
        assert len(probabilities) == len(oplist)
        self.oplist = oplist
        self.rho = np.cumsum(probabilities)
        self.rng = rng

    def __get_index__(self):
        v = self.rng.rand() * self.rho[-1]
        for i in range(len(self.rho)):
            if self.rho[i] > v:
                return i

    def get_new_individual(self, candidate_list):
        """Choose operator and use it on the candidate. """
        to_use = self.__get_index__()
        return self.oplist[to_use].get_new_individual(candidate_list)

    def get_operator(self):
        """Choose operator and return it."""
        to_use = self.__get_index__()
        return self.oplist[to_use]