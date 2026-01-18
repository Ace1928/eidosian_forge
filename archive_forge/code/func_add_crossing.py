import random
from .links_base import CrossingEntryPoint
from ..sage_helper import _within_sage
def add_crossing(self, crossing):
    indices = self.strand_indices
    t = self.ring.gen()
    a, b = entry_pts_ab(crossing)
    n = self.A.nrows()
    assert indices[a] == n and indices[b] == n + 1
    T = t if crossing.sign == 1 else t ** (-1)
    B = matrix([[1, 1 - T], [0, T]])
    self.A = block_diagonal_matrix([self.A, B])