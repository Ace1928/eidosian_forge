from dataclasses import dataclass, field
from enum import Enum
from heapq import heappop, heappush
from itertools import count
from math import isnan
from operator import itemgetter
from queue import PriorityQueue
import networkx as nx
from networkx.utils import UnionFind, not_implemented_for, py_random_state
class SpanningTreeIterator:
    """
    Iterate over all spanning trees of a graph in either increasing or
    decreasing cost.

    Notes
    -----
    This iterator uses the partition scheme from [1]_ (included edges,
    excluded edges and open edges) as well as a modified Kruskal's Algorithm
    to generate minimum spanning trees which respect the partition of edges.
    For spanning trees with the same weight, ties are broken arbitrarily.

    References
    ----------
    .. [1] G.K. Janssens, K. Sörensen, An algorithm to generate all spanning
           trees in order of increasing cost, Pesquisa Operacional, 2005-08,
           Vol. 25 (2), p. 219-229,
           https://www.scielo.br/j/pope/a/XHswBwRwJyrfL88dmMwYNWp/?lang=en
    """

    @dataclass(order=True)
    class Partition:
        """
        This dataclass represents a partition and stores a dict with the edge
        data and the weight of the minimum spanning tree of the partition dict.
        """
        mst_weight: float
        partition_dict: dict = field(compare=False)

        def __copy__(self):
            return SpanningTreeIterator.Partition(self.mst_weight, self.partition_dict.copy())

    def __init__(self, G, weight='weight', minimum=True, ignore_nan=False):
        """
        Initialize the iterator

        Parameters
        ----------
        G : nx.Graph
            The directed graph which we need to iterate trees over

        weight : String, default = "weight"
            The edge attribute used to store the weight of the edge

        minimum : bool, default = True
            Return the trees in increasing order while true and decreasing order
            while false.

        ignore_nan : bool, default = False
            If a NaN is found as an edge weight normally an exception is raised.
            If `ignore_nan is True` then that edge is ignored instead.
        """
        self.G = G.copy()
        self.weight = weight
        self.minimum = minimum
        self.ignore_nan = ignore_nan
        self.partition_key = 'SpanningTreeIterators super secret partition attribute name'

    def __iter__(self):
        """
        Returns
        -------
        SpanningTreeIterator
            The iterator object for this graph
        """
        self.partition_queue = PriorityQueue()
        self._clear_partition(self.G)
        mst_weight = partition_spanning_tree(self.G, self.minimum, self.weight, self.partition_key, self.ignore_nan).size(weight=self.weight)
        self.partition_queue.put(self.Partition(mst_weight if self.minimum else -mst_weight, {}))
        return self

    def __next__(self):
        """
        Returns
        -------
        (multi)Graph
            The spanning tree of next greatest weight, which ties broken
            arbitrarily.
        """
        if self.partition_queue.empty():
            del self.G, self.partition_queue
            raise StopIteration
        partition = self.partition_queue.get()
        self._write_partition(partition)
        next_tree = partition_spanning_tree(self.G, self.minimum, self.weight, self.partition_key, self.ignore_nan)
        self._partition(partition, next_tree)
        self._clear_partition(next_tree)
        return next_tree

    def _partition(self, partition, partition_tree):
        """
        Create new partitions based of the minimum spanning tree of the
        current minimum partition.

        Parameters
        ----------
        partition : Partition
            The Partition instance used to generate the current minimum spanning
            tree.
        partition_tree : nx.Graph
            The minimum spanning tree of the input partition.
        """
        p1 = self.Partition(0, partition.partition_dict.copy())
        p2 = self.Partition(0, partition.partition_dict.copy())
        for e in partition_tree.edges:
            if e not in partition.partition_dict:
                p1.partition_dict[e] = EdgePartition.EXCLUDED
                p2.partition_dict[e] = EdgePartition.INCLUDED
                self._write_partition(p1)
                p1_mst = partition_spanning_tree(self.G, self.minimum, self.weight, self.partition_key, self.ignore_nan)
                p1_mst_weight = p1_mst.size(weight=self.weight)
                if nx.is_connected(p1_mst):
                    p1.mst_weight = p1_mst_weight if self.minimum else -p1_mst_weight
                    self.partition_queue.put(p1.__copy__())
                p1.partition_dict = p2.partition_dict.copy()

    def _write_partition(self, partition):
        """
        Writes the desired partition into the graph to calculate the minimum
        spanning tree.

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
                d[self.partition_key] = EdgePartition.OPEN

    def _clear_partition(self, G):
        """
        Removes partition data from the graph
        """
        for u, v, d in G.edges(data=True):
            if self.partition_key in d:
                del d[self.partition_key]