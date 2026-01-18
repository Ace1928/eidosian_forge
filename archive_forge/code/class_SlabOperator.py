from operator import itemgetter
from collections import Counter
from itertools import permutations
import numpy as np
from ase.ga.offspring_creator import OffspringCreator
from ase.ga.element_mutations import get_periodic_table_distance
from ase.utils import atoms_to_spglib_cell
class SlabOperator(OffspringCreator):

    def __init__(self, verbose=False, num_muts=1, allowed_compositions=None, distribution_correction_function=None, element_pools=None, rng=np.random):
        OffspringCreator.__init__(self, verbose, num_muts=num_muts, rng=rng)
        self.allowed_compositions = allowed_compositions
        self.element_pools = element_pools
        if distribution_correction_function is None:
            self.dcf = dummy_func
        else:
            self.dcf = distribution_correction_function

    def get_symbols_to_use(self, syms):
        """Get the symbols to use for the offspring candidate. The returned
        list of symbols will respect self.allowed_compositions"""
        if self.allowed_compositions is None:
            return syms
        unique_syms, counts = np.unique(syms, return_counts=True)
        comp, unique_syms = zip(*sorted(zip(counts, unique_syms), reverse=True))
        for cc in self.allowed_compositions:
            comp += (0,) * (len(cc) - len(comp))
            if comp == tuple(sorted(cc)):
                return syms
        comp_diff = self.get_closest_composition_diff(comp)
        to_add, to_rem = get_add_remove_lists(**dict(zip(unique_syms, comp_diff)))
        for add, rem in zip(to_add, to_rem):
            tbc = [i for i in range(len(syms)) if syms[i] == rem]
            ai = self.rng.choice(tbc)
            syms[ai] = add
        return syms

    def get_add_remove_elements(self, syms):
        if self.element_pools is None or self.allowed_compositions is None:
            return ([], [])
        unique_syms, pool_number, comp = get_ordered_composition(syms, self.element_pools)
        stay_comp, stay_syms = ([], [])
        add_rem = {}
        per_pool = len(self.allowed_compositions[0]) / len(self.element_pools)
        pool_count = np.zeros(len(self.element_pools), dtype=int)
        for pn, num, sym in zip(pool_number, comp, unique_syms):
            pool_count[pn] += 1
            if pool_count[pn] <= per_pool:
                stay_comp.append(num)
                stay_syms.append(sym)
            else:
                add_rem[sym] = -num
        diff = self.get_closest_composition_diff(stay_comp)
        add_rem.update(dict(((s, c) for s, c in zip(stay_syms, diff))))
        return get_add_remove_lists(**add_rem)

    def get_closest_composition_diff(self, c):
        comp = np.array(c)
        mindiff = 10000000000.0
        allowed_list = list(self.allowed_compositions)
        self.rng.shuffle(allowed_list)
        for ac in allowed_list:
            diff = self.get_composition_diff(comp, ac)
            numdiff = sum([abs(i) for i in diff])
            if numdiff < mindiff:
                mindiff = numdiff
                ccdiff = diff
        return ccdiff

    def get_composition_diff(self, c1, c2):
        difflen = len(c1) - len(c2)
        if difflen > 0:
            c2 += (0,) * difflen
        return np.array(c2) - c1

    def get_possible_mutations(self, a):
        unique_syms, comp = np.unique(sorted(a.get_chemical_symbols()), return_counts=True)
        min_num = min([i for i in np.ravel(list(self.allowed_compositions)) if i > 0])
        muts = set()
        for i, n in enumerate(comp):
            if n != 0:
                muts.add((unique_syms[i], n))
            if n % min_num >= 0:
                for j in range(1, n // min_num):
                    muts.add((unique_syms[i], min_num * j))
        return list(muts)

    def get_all_element_mutations(self, a):
        """Get all possible mutations for the supplied atoms object given
        the element pools."""
        muts = []
        symset = set(a.get_chemical_symbols())
        for sym in symset:
            for pool in self.element_pools:
                if sym in pool:
                    muts.extend([(sym, s) for s in pool if s not in symset])
        return muts

    def finalize_individual(self, indi):
        atoms_string = ''.join(indi.get_chemical_symbols())
        indi.info['key_value_pairs']['atoms_string'] = atoms_string
        return OffspringCreator.finalize_individual(self, indi)