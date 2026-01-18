from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
class CutSpliceSlabCrossover(SlabOperator):

    def __init__(self, allowed_compositions=None, element_pools=None, verbose=False, num_muts=1, tries=1000, min_ratio=0.25, distribution_correction_function=None, rng=np.random):
        SlabOperator.__init__(self, verbose, num_muts, allowed_compositions, distribution_correction_function, element_pools=element_pools, rng=rng)
        self.tries = tries
        self.min_ratio = min_ratio
        self.descriptor = 'CutSpliceSlabCrossover'

    def get_new_individual(self, parents):
        f, m = parents
        indi = self.initialize_individual(f, self.operate(f, m))
        indi.info['data']['parents'] = [i.info['confid'] for i in parents]
        parent_message = ': Parents {0} {1}'.format(f.info['confid'], m.info['confid'])
        return (self.finalize_individual(indi), self.descriptor + parent_message)

    def operate(self, f, m):
        child = f.copy()
        fp = f.positions
        ma = np.max(fp.transpose(), axis=1)
        mi = np.min(fp.transpose(), axis=1)
        for _ in range(self.tries):
            rv = [self.rng.rand() for _ in range(3)]
            midpoint = (ma - mi) * rv + mi
            theta = self.rng.rand() * 2 * np.pi
            phi = self.rng.rand() * np.pi
            e = np.array((np.sin(phi) * np.cos(theta), np.sin(theta) * np.sin(phi), np.cos(phi)))
            d2fp = np.dot(fp - midpoint, e)
            fpart = d2fp > 0
            ratio = float(np.count_nonzero(fpart)) / len(f)
            if ratio < self.min_ratio or ratio > 1 - self.min_ratio:
                continue
            syms = np.where(fpart, f.get_chemical_symbols(), m.get_chemical_symbols())
            dists2plane = abs(d2fp)
            to_add, to_rem = self.get_add_remove_elements(syms)
            for add, rem in zip(to_add, to_rem):
                tbc = [(dists2plane[i], i) for i in range(len(syms)) if syms[i] == rem]
                ai = sorted(tbc)[0][1]
                syms[ai] = add
            child.set_chemical_symbols(syms)
            break
        self.dcf(child)
        return child