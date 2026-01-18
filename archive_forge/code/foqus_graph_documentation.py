import copy, logging
from pyomo.common.dependencies import numpy

        Returns an adjacency list and a reverse adjacency list
        of node indexes for a MultiDiGraph.

        Arguments
        ---------
            G
                A networkx MultiDiGraph
            excludeEdges
                List of edge indexes to ignore when considering neighbors
            nodes
                List of nodes to form the adjacencies from
            multi
                If True, adjacency lists will contains tuples of
                (node, key) for every edge between two nodes

        Returns
        -------
            i2n
                Map from index to node for all nodes included in nodes
            adj
                Adjacency list of successor indexes
            adjR
                Reverse adjacency list of predecessor indexes
        