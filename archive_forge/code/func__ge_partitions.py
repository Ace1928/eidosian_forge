import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@property
def _ge_partitions(self):
    if self._ge_partitions_ is None:

        def edgematch(edge1, edge2):
            return self.edge_equality(self.graph, edge1, self.graph, edge2)
        self._ge_partitions_ = make_partitions(self.graph.edges, edgematch)
    return self._ge_partitions_