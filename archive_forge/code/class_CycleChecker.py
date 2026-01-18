import itertools
import numpy as np
from ase.utils import pbc2pbc
from ase.cell import Cell
class CycleChecker:

    def __init__(self, d):
        assert d in [2, 3]
        n = {2: 6, 3: 12}[d]
        max_cycle_length = np.prod([n - i for i in range(d)]) * np.prod(d)
        self.visited = np.zeros((max_cycle_length, 3 * d), dtype=int)

    def add_site(self, H):
        H = H.ravel()
        found = (self.visited == H).all(axis=1).any()
        self.visited = np.roll(self.visited, 1, axis=0)
        self.visited[0] = H
        return found