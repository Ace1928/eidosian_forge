import numpy as np
from ase import Atoms
class OffspringCreator:
    """Base class for all procreation operators

    Parameters:

    verbose: Be verbose and print some stuff

    rng: Random number generator
        By default numpy.random.
    """

    def __init__(self, verbose=False, num_muts=1, rng=np.random):
        self.descriptor = 'OffspringCreator'
        self.verbose = verbose
        self.min_inputs = 0
        self.num_muts = num_muts
        self.rng = rng

    def get_min_inputs(self):
        """Returns the number of inputs required for a mutation,
        this is to know how many candidates should be selected
        from the population."""
        return self.min_inputs

    def get_new_individual(self, parents):
        """Function that returns a new individual.
        Overwrite in subclass."""
        raise NotImplementedError

    def finalize_individual(self, indi):
        """Call this function just before returning the new individual"""
        indi.info['key_value_pairs']['origin'] = self.descriptor
        return indi

    @classmethod
    def initialize_individual(cls, parent, indi=None):
        """Initializes a new individual that inherits some parameters
        from the parent, and initializes the info dictionary.
        If the new individual already has more structure it can be
        supplied in the parameter indi."""
        if indi is None:
            indi = Atoms(pbc=parent.get_pbc(), cell=parent.get_cell())
        else:
            indi = indi.copy()
        indi.info['key_value_pairs'] = {'extinct': 0}
        indi.info['data'] = {}
        return indi