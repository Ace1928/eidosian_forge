from math import log10, atan2, cos, sin
from ase.build import hcp0001, fcc111, bcc111
import numpy as np
def _root_surface_analysis(primitive_slab, root, eps=1e-08):
    """A tool to analyze a slab and look for valid roots that exist, up to
    the given root. This is useful for generating all possible cells
    without prior knowledge.

    *primitive slab* is the primitive cell to analyze.

    *root* is the desired root to find, and all below.

    This is the internal function which gives extra data to root_surface.
    """
    logeps = int(-log10(eps))
    xscale, cell_vectors = _root_cell_normalization(primitive_slab)
    points = np.indices((root + 1, root + 1)).T.reshape(-1, 2)
    cell_points = [cell_vectors[0] * x + cell_vectors[1] * y for x, y in points]
    roots = np.around(np.linalg.norm(cell_points, axis=1) ** 2, logeps)
    valid_roots = np.nonzero(roots == root)[0]
    if len(valid_roots) == 0:
        raise ValueError('Invalid root {} for cell {}'.format(root, cell_vectors))
    int_roots = np.array([int(this_root) for this_root in roots if this_root.is_integer() and this_root <= root])
    return (cell_points, cell_points[np.nonzero(roots == root)[0][0]], set(int_roots[1:]))