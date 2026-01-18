import numpy as np
import itertools
from scipy import sparse as sp
from scipy.spatial import cKDTree
import scipy.sparse.csgraph as csgraph
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import complete_cell, find_mic, wrap_positions
from ase.geometry import minkowski_reduce
from ase.cell import Cell
class NewPrimitiveNeighborList:
    """Neighbor list object. Wrapper around neighbor_list and first_neighbors.

    cutoffs: list of float
        List of cutoff radii - one for each atom. If the spheres (defined by
        their cutoff radii) of two atoms overlap, they will be counted as
        neighbors.
    skin: float
        If no atom has moved more than the skin-distance since the
        last call to the :meth:`~ase.neighborlist.NewPrimitiveNeighborList.update()`
        method, then the neighbor list can be reused. This will save
        some expensive rebuilds of the list, but extra neighbors outside
        the cutoff will be returned.
    sorted: bool
        Sort neighbor list.
    self_interaction: bool
        Should an atom return itself as a neighbor?
    bothways: bool
        Return all neighbors.  Default is to return only "half" of
        the neighbors.

    Example::

      nl = NeighborList([2.3, 1.7])
      nl.update(atoms)
      indices, offsets = nl.get_neighbors(0)
    """

    def __init__(self, cutoffs, skin=0.3, sorted=False, self_interaction=True, bothways=False, use_scaled_positions=False):
        self.cutoffs = np.asarray(cutoffs) + skin
        self.skin = skin
        self.sorted = sorted
        self.self_interaction = self_interaction
        self.bothways = bothways
        self.nupdates = 0
        self.use_scaled_positions = use_scaled_positions
        self.nneighbors = 0
        self.npbcneighbors = 0

    def update(self, pbc, cell, positions, numbers=None):
        """Make sure the list is up to date."""
        if self.nupdates == 0:
            self.build(pbc, cell, positions, numbers=numbers)
            return True
        if (self.pbc != pbc).any() or (self.cell != cell).any() or ((self.positions - positions) ** 2).sum(1).max() > self.skin ** 2:
            self.build(pbc, cell, positions, numbers=numbers)
            return True
        return False

    def build(self, pbc, cell, positions, numbers=None):
        """Build the list.
        """
        self.pbc = np.array(pbc, copy=True)
        self.cell = np.array(cell, copy=True)
        self.positions = np.array(positions, copy=True)
        pair_first, pair_second, offset_vec = primitive_neighbor_list('ijS', pbc, cell, positions, self.cutoffs, numbers=numbers, self_interaction=self.self_interaction, use_scaled_positions=self.use_scaled_positions)
        if len(positions) > 0 and (not self.bothways):
            offset_x, offset_y, offset_z = offset_vec.T
            mask = offset_z > 0
            mask &= offset_y == 0
            mask |= offset_y > 0
            mask &= offset_x == 0
            mask |= offset_x > 0
            mask |= (pair_first <= pair_second) & (offset_vec == 0).all(axis=1)
            pair_first = pair_first[mask]
            pair_second = pair_second[mask]
            offset_vec = offset_vec[mask]
        if len(positions) > 0 and self.sorted:
            mask = np.argsort(pair_first * len(pair_first) + pair_second)
            pair_first = pair_first[mask]
            pair_second = pair_second[mask]
            offset_vec = offset_vec[mask]
        self.pair_first = pair_first
        self.pair_second = pair_second
        self.offset_vec = offset_vec
        self.first_neigh = first_neighbors(len(positions), pair_first)
        self.nupdates += 1

    def get_neighbors(self, a):
        """Return neighbors of atom number a.

        A list of indices and offsets to neighboring atoms is
        returned.  The positions of the neighbor atoms can be
        calculated like this:

        >>>  indices, offsets = nl.get_neighbors(42)
        >>>  for i, offset in zip(indices, offsets):
        >>>      print(atoms.positions[i] + dot(offset, atoms.get_cell()))

        Notice that if get_neighbors(a) gives atom b as a neighbor,
        then get_neighbors(b) will not return a as a neighbor - unless
        bothways=True was used."""
        return (self.pair_second[self.first_neigh[a]:self.first_neigh[a + 1]], self.offset_vec[self.first_neigh[a]:self.first_neigh[a + 1]])