import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@property
def _gn_partitions(self):
    if self._gn_partitions_ is None:

        def nodematch(node1, node2):
            return self.node_equality(self.graph, node1, self.graph, node2)
        self._gn_partitions_ = make_partitions(self.graph.nodes, nodematch)
    return self._gn_partitions_