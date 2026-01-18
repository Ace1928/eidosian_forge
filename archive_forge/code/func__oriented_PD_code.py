import copy
import re
import snappy_manifolds
from collections import OrderedDict, namedtuple
from .. import graphs
from .ordered_set import OrderedSet
def _oriented_PD_code(self, KnotTheory=False, min_strand_index=0):
    PD = {c: [-1, -1, -1, -1] for c in self.crossings}
    label = min_strand_index
    for comp in self.link_components:
        for cep in comp:
            op_cep = cep.opposite()
            PD[cep.crossing][cep.strand_index] = label
            PD[op_cep.crossing][op_cep.strand_index] = label
            label += 1
    PD = PD.values()
    if KnotTheory:
        PD = 'PD' + repr(PD).replace('[', 'X[')[1:]
    else:
        PD = [tuple(x) for x in PD]
    return PD