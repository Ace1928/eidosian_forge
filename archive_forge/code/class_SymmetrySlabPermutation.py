from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
class SymmetrySlabPermutation(SlabOperator):
    """Permutes the atoms in the slab until it has a higher symmetry number."""

    def __init__(self, verbose=False, num_muts=1, sym_goal=100, max_tries=50, allowed_compositions=None, distribution_correction_function=None, rng=np.random):
        SlabOperator.__init__(self, verbose, num_muts, allowed_compositions, distribution_correction_function, rng=rng)
        if spglib is None:
            print('SymmetrySlabPermutation needs spglib to function')
        assert sym_goal >= 1
        self.sym_goal = sym_goal
        self.max_tries = max_tries
        self.descriptor = 'SymmetrySlabPermutation'

    def get_new_individual(self, parents):
        f = parents[0]
        if len(set(f.get_chemical_symbols())) == 1:
            f = parents[1]
            if len(set(f.get_chemical_symbols())) == 1:
                return (None, '{1} not possible in {0}'.format(f.info['confid'], self.descriptor))
        indi = self.initialize_individual(f, self.operate(f))
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]
        parent_message = ': Parent {0}'.format(f.info['confid'])
        return (self.finalize_individual(indi), self.descriptor + parent_message)

    def operate(self, atoms):
        sym_num = 1
        sg = self.sym_goal
        while sym_num < sg:
            for _ in range(self.max_tries):
                for _ in range(2):
                    permute2(atoms, rng=self.rng)
                self.dcf(atoms)
                sym_num = spglib.get_symmetry_dataset(atoms_to_spglib_cell(atoms))['number']
                if sym_num >= sg:
                    break
            sg -= 1
        return atoms