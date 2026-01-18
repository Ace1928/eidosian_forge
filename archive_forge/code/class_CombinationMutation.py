import numpy as np
from ase import Atoms
class CombinationMutation(OffspringCreator):
    """Combine two or more mutations into one operation.

    Parameters:

    mutations: Operator instances
        Supply two or more mutations that will applied one after the other
        as one mutation operation. The order of the supplied mutations prevail
        when applying the mutations.

    """

    def __init__(self, *mutations, verbose=False):
        super(CombinationMutation, self).__init__(verbose=verbose)
        self.descriptor = 'CombinationMutation'
        msg = 'Too few operators supplied to a CombinationMutation'
        assert len(mutations) > 1, msg
        self.operators = mutations

    def get_new_individual(self, parents):
        f = parents[0]
        indi = self.mutate(f)
        if indi is None:
            return (indi, 'mutation: {}'.format(self.descriptor))
        indi = self.initialize_individual(f, indi)
        indi.info['data']['parents'] = [f.info['confid']]
        return (self.finalize_individual(indi), 'mutation: {}'.format(self.descriptor))

    def mutate(self, atoms):
        """Perform the mutations one at a time."""
        for op in self.operators:
            if atoms is not None:
                atoms = op.mutate(atoms)
        return atoms