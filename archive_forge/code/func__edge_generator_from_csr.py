import networkx as nx
import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
import math
from .. import measure, segmentation, util, color
from .._shared.version_requirements import require
def _edge_generator_from_csr(csr_matrix):
    """Yield weighted edge triples for use by NetworkX from a CSR matrix.

    This function is a straight rewrite of
    `networkx.convert_matrix._csr_gen_triples`. Since that is a private
    function, it is safer to include our own here.

    Parameters
    ----------
    csr_matrix : scipy.sparse.csr_matrix
        The input matrix. An edge (i, j, w) will be yielded if there is a
        data value for coordinates (i, j) in the matrix, even if that value
        is 0.

    Yields
    ------
    i, j, w : (int, int, float) tuples
        Each value `w` in the matrix along with its coordinates (i, j).

    Examples
    --------

    >>> dense = np.eye(2, dtype=float)
    >>> csr = sparse.csr_matrix(dense)
    >>> edges = _edge_generator_from_csr(csr)
    >>> list(edges)
    [(0, 0, 1.0), (1, 1, 1.0)]
    """
    nrows = csr_matrix.shape[0]
    values = csr_matrix.data
    indptr = csr_matrix.indptr
    col_indices = csr_matrix.indices
    for i in range(nrows):
        for j in range(indptr[i], indptr[i + 1]):
            yield (i, col_indices[j], values[j])