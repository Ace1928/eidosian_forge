from math import tanh, sqrt, exp
from operator import itemgetter
import numpy as np
from ase.db.core import now
from ase.ga import get_raw_score
def __get_fitness__(self, candidates):
    msg = 'This is a multi-objective fitness function'
    msg += ' so there must be at least two datasets'
    msg += ' stated in the rank_data and abs_data variables'
    assert len(self.rank_data) + len(self.abs_data) >= 2, msg
    expf = self.exp_function
    all_fitnesses = []
    used = set()
    for rd in self.rank_data:
        used.add(rd)
        all_fitnesses.append(self.get_rank(candidates, key=rd))
    for d in self.abs_data:
        if d not in used:
            used.add(d)
            all_fitnesses.append(self.get_nonrank(candidates, key=d))
    fordered = list(zip(range(len(all_fitnesses[0])), *all_fitnesses))
    mvf_rank = -1
    rec_vrc = []
    mvf_list = []
    fordered.sort(key=itemgetter(1), reverse=True)
    for a in fordered:
        order, rest = (a[0], a[1:])
        if order not in rec_vrc:
            pff = []
            pff2 = []
            rec_vrc.append(order)
            pff.append((order, rest))
            for b in fordered:
                border, brest = (b[0], b[1:])
                if border not in rec_vrc:
                    if np.any(np.array(brest) >= np.array(rest)):
                        pff.append((border, brest))
            for na in pff:
                norder, nrest = (na[0], na[1:])
                dom = False
                for nb in pff:
                    nborder, nbrest = (nb[0], nb[1:])
                    if norder != nborder:
                        if np.all(np.array(nbrest) > np.array(nrest)):
                            dom = True
                if not dom:
                    pff2.append((norder, nrest))
            for ffa in pff2:
                fforder, ffrest = (ffa[0], ffa[1:])
                rec_vrc.append(fforder)
                mvf_list.append((fforder, mvf_rank, ffrest))
            mvf_rank = mvf_rank - 1
    mvf_list.sort(key=itemgetter(0), reverse=False)
    rfro = np.array(list(zip(*mvf_list))[1])
    if not expf:
        rmax = max(rfro)
        rmin = min(rfro)
        T = rmin - rmax
        msg = 'Equal fitness for best and worst candidate in the '
        msg += 'population! Fitness scaling is impossible! '
        msg += 'Try with a larger population.'
        assert T != 0.0, msg
        return 0.5 * (1.0 - np.tanh(2.0 * (rfro - rmax) / T - 1.0))
    else:
        return self.exp_prefactor ** (-rfro - 1)