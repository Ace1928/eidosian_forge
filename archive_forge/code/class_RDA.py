import numpy as np
from collections import defaultdict
from ase.geometry.dimensionality.disjoint_set import DisjointSet
class RDA:

    def __init__(self, num_atoms):
        """
        Initializes the RDA class.

        A disjoint set is used to maintain the component graph.

        Parameters:

        num_atoms: int    The number of atoms in the unit cell.
        """
        self.bonds = []
        self.graph = DisjointSet(num_atoms)
        self.adjacency = None
        self.hcached = None
        self.components_cached = None
        self.cdim_cached = None

    def insert_bond(self, i, j, offset):
        """
        Adds a bond to the list of graph edges.

        Graph components are merged if the bond does not cross a cell boundary.
        Bonds which cross cell boundaries can inappropriately connect
        components which are not connected in the infinite crystal.  This is
        tested during graph traversal.

        Parameters:

        i: int           The index of the first atom.
        n: int           The index of the second atom.
        offset: tuple    The cell offset of the second atom.
        """
        roffset = tuple(-np.array(offset))
        if offset == (0, 0, 0):
            self.graph.union(i, j)
        else:
            self.bonds += [(i, j, offset)]
            self.bonds += [(j, i, roffset)]

    def check(self):
        """
        Determines the dimensionality histogram.

        The component graph is traversed (using BFS) until the matrix rank
        of the subspace spanned by the visited components no longer increases.

        Returns:
        hist : tuple         Dimensionality histogram.
        """
        adjacency = build_adjacency_list(self.graph.find_all(), self.bonds)
        if adjacency == self.adjacency:
            return self.hcached
        self.adjacency = adjacency
        self.all_visited, self.ranks = traverse_component_graphs(adjacency)
        res = merge_mutual_visits(self.all_visited, self.ranks, self.graph)
        _, self.all_visited, self.ranks = res
        self.roots = np.unique(self.graph.find_all())
        h = get_dimensionality_histogram(self.ranks, self.roots)
        self.hcached = h
        return h

    def get_components(self):
        """
        Determines the dimensionality and constituent atoms of each component.

        Returns:
        components: array    The component ID of every atom
        """
        component_dim = {e: self.ranks[e] for e in self.roots}
        relabelled_components = self.graph.find_all(relabel=True)
        relabelled_dim = {}
        for k, v in component_dim.items():
            relabelled_dim[relabelled_components[k]] = v
        self.cdim_cached = relabelled_dim
        self.components_cached = relabelled_components
        return (relabelled_components, relabelled_dim)