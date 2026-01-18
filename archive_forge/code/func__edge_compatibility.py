import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@property
def _edge_compatibility(self):
    if self._edge_compat_ is not None:
        return self._edge_compat_
    self._edge_compat_ = {}
    for sge_part_color, ge_part_color in itertools.product(range(len(self._sge_partitions)), range(len(self._ge_partitions))):
        sge = next(iter(self._sge_partitions[sge_part_color]))
        ge = next(iter(self._ge_partitions[ge_part_color]))
        if self.edge_equality(self.subgraph, sge, self.graph, ge):
            self._edge_compat_[sge_part_color] = ge_part_color
    return self._edge_compat_