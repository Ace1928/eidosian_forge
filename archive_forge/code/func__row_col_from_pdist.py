import itertools
import numpy as np
from ase.geometry import complete_cell
from ase.geometry.minkowski_reduction import minkowski_reduce
from ase.utils import pbc2pbc
from ase.cell import Cell
def _row_col_from_pdist(dim, i):
    """Calculate the i,j index in the square matrix for an index in a
    condensed (triangular) matrix.
    """
    i = np.array(i)
    b = 1 - 2 * dim
    x = np.floor((-b - np.sqrt(b ** 2 - 8 * i)) / 2).astype(int)
    y = (i + x * (b + x + 2) / 2 + 1).astype(int)
    if i.shape:
        return list(zip(x, y))
    else:
        return [(x, y)]