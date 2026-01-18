import string
from dataclasses import dataclass, field
from enum import Enum
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import py_random_state
from .recognition import is_arborescence, is_branching
def _write_partition(self, partition):
    """
        Writes the desired partition into the graph to calculate the minimum
        spanning tree. Also, if one incoming edge is included, mark all others
        as excluded so that if that vertex is merged during Edmonds' algorithm
        we cannot still pick another of that vertex's included edges.

        Parameters
        ----------
        partition : Partition
            A Partition dataclass describing a partition on the edges of the
            graph.
        """
    for u, v, d in self.G.edges(data=True):
        if (u, v) in partition.partition_dict:
            d[self.partition_key] = partition.partition_dict[u, v]
        else:
            d[self.partition_key] = nx.EdgePartition.OPEN
    for n in self.G:
        included_count = 0
        excluded_count = 0
        for u, v, d in self.G.in_edges(nbunch=n, data=True):
            if d.get(self.partition_key) == nx.EdgePartition.INCLUDED:
                included_count += 1
            elif d.get(self.partition_key) == nx.EdgePartition.EXCLUDED:
                excluded_count += 1
        if included_count == 1 and excluded_count != self.G.in_degree(n) - 1:
            for u, v, d in self.G.in_edges(nbunch=n, data=True):
                if d.get(self.partition_key) != nx.EdgePartition.INCLUDED:
                    d[self.partition_key] = nx.EdgePartition.EXCLUDED