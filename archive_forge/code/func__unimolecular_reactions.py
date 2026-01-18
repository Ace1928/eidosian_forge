import math
from collections import OrderedDict, defaultdict
from itertools import chain
from .chemistry import Reaction, Substance
from .units import to_unitless
from .util.pyutil import deprecated
def _unimolecular_reactions(self):
    A = [None] * self.ns
    unconsidered_ri = set()
    for i, r in enumerate(self.rxns):
        if r.order() == 1:
            keys = [k for k, v in r.reac.items() if v != 0]
            if len(keys) == 1:
                ri = self.as_substance_index(keys[0])
            else:
                raise NotImplementedError('Need 1 or 2 keys')
            if A[ri] is None:
                A[ri] = list()
            A[ri].append((i, r))
        else:
            unconsidered_ri.add(i)
    return (A, unconsidered_ri)