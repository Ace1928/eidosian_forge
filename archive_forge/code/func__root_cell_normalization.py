from math import log10, atan2, cos, sin
from ase.build import hcp0001, fcc111, bcc111
import numpy as np
def _root_cell_normalization(primitive_slab):
    """Returns the scaling factor for x axis and cell normalized by that factor"""
    xscale = np.linalg.norm(primitive_slab.cell[0, 0:2])
    cell_vectors = primitive_slab.cell[0:2, 0:2] / xscale
    return (xscale, cell_vectors)