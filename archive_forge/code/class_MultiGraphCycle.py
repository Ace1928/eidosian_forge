from __future__ import annotations
import itertools
import operator
from typing import TYPE_CHECKING
import networkx as nx
import numpy as np
from monty.json import MSONable
class MultiGraphCycle(MSONable):
    """Class used to describe a cycle in a multigraph.

    nodes are the nodes of the cycle and edge_indices are the indices of the edges in the cycle.
    The nth index in edge_indices corresponds to the edge index between the nth node in nodes and
    the (n+1)th node in nodes with the exception of the last one being the edge index between
    the last node in nodes and the first node in nodes

    Example: A cycle
        nodes:          1 - 3 - 4 - 0 - 2 - (1)
        edge_indices:     0 . 1 . 0 . 2 . 0 . (0)
    """

    def __init__(self, nodes, edge_indices, validate=True, ordered=None):
        """
        Args:
            nodes:
            edge_indices:
            validate:
            ordered:
        """
        self.nodes = tuple(nodes)
        self.edge_indices = tuple(edge_indices)
        if validate:
            self.validate()
        if ordered is not None:
            self.ordered = ordered
        else:
            self.order()
        self.edge_deltas = self.per = None

    def _is_valid(self, check_strict_ordering=False):
        """Check if a MultiGraphCycle is valid.

        This method checks that:
        1. there are no duplicate nodes,
        2. there are either 1 or more than 2 nodes

        Returns:
            bool: True if the SimpleGraphCycle is valid.
        """
        if len(self.nodes) != len(self.edge_indices):
            return (False, 'Number of nodes different from number of edge indices.')
        if len(self.nodes) == 0:
            return (False, 'Empty cycle is not valid.')
        if len(self.nodes) != len(set(self.nodes)):
            return (False, 'Duplicate nodes.')
        if len(self.nodes) == 2 and self.edge_indices[0] == self.edge_indices[1]:
            return (False, 'Cycles with two nodes cannot use the same edge for the cycle.')
        if check_strict_ordering:
            try:
                sorted_nodes = sorted(self.nodes)
            except TypeError as te:
                msg = te.args[0]
                if "'<' not supported between instances of" in msg:
                    return (False, 'The nodes are not sortable.')
                raise
            res = all((i < j for i, j in zip(sorted_nodes, sorted_nodes[1:])))
            if not res:
                return (False, 'The list of nodes in the cycle cannot be strictly ordered.')
        return (True, '')

    def validate(self, check_strict_ordering=False):
        """
        Args:
            check_strict_ordering:
        """
        is_valid, msg = self._is_valid(check_strict_ordering=check_strict_ordering)
        if not is_valid:
            raise ValueError(f'MultiGraphCycle is not valid : {msg}')

    def order(self, raise_on_fail: bool=True):
        """Orders the SimpleGraphCycle.

        The ordering is performed such that the first node is the "lowest" one
        and the second node is the lowest one of the two neighbor nodes of the
        first node. If raise_on_fail is set to True a RuntimeError will be
        raised if the ordering fails.

        Args:
            raise_on_fail: If set to True, will raise a RuntimeError if the ordering fails.
        """
        try:
            self.validate(check_strict_ordering=True)
        except ValueError as ve:
            msg = ve.args[0]
            if 'MultiGraphCycle is not valid :' in msg and (not raise_on_fail):
                self.ordered = False
                return
            raise
        if len(self.nodes) == 1:
            self.ordered = True
            return
        node_classes = {n.__class__ for n in self.nodes}
        if len(node_classes) > 1:
            if raise_on_fail:
                raise ValueError('Could not order simple graph cycle as the nodes are of different classes.')
            self.ordered = False
            return
        min_index, _min_node = min(enumerate(self.nodes), key=operator.itemgetter(1))
        if len(self.nodes) == 2:
            self.nodes = tuple(sorted(self.nodes))
            self.edge_indices = tuple(sorted(self.edge_indices))
            self.ordered = True
            return
        reverse = self.nodes[(min_index - 1) % len(self.nodes)] < self.nodes[(min_index + 1) % len(self.nodes)]
        if reverse:
            self.nodes = self.nodes[min_index::-1] + self.nodes[:min_index:-1]
            min_edge_index = (min_index - 1) % len(self.nodes)
            self.edge_indices = self.edge_indices[min_edge_index::-1] + self.edge_indices[:min_edge_index:-1]
        else:
            self.nodes = self.nodes[min_index:] + self.nodes[:min_index]
            self.edge_indices = self.edge_indices[min_index:] + self.edge_indices[:min_index]
        self.ordered = True

    def __hash__(self) -> int:
        return len(self.nodes)

    def __len__(self):
        return len(self.nodes)

    def __str__(self):
        out = ['Multigraph cycle with nodes :']
        cycle = []
        for inode, node1, node2 in zip(itertools.count(), self.nodes[:-1], self.nodes[1:]):
            cycle.append(f'{node1} -*{self.edge_indices[inode]}*- {node2}')
        cycle.append(f'{self.nodes[-1]} -*{self.edge_indices[-1]}*- {self.nodes[0]}')
        out.extend(cycle)
        return '\n'.join(out)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MultiGraphCycle):
            return NotImplemented
        if not self.ordered or not other.ordered:
            raise RuntimeError('Multigraph cycles should be ordered in order to be compared.')
        return self.nodes == other.nodes and self.edge_indices == other.edge_indices