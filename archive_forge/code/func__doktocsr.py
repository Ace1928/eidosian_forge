from sympy.core.containers import Dict
from sympy.core.symbol import Dummy
from sympy.utilities.iterables import is_sequence
from sympy.utilities.misc import as_int, filldedent
from .sparse import MutableSparseMatrix as SparseMatrix
def _doktocsr(dok):
    """Converts a sparse matrix to Compressed Sparse Row (CSR) format.

    Parameters
    ==========

    A : contains non-zero elements sorted by key (row, column)
    JA : JA[i] is the column corresponding to A[i]
    IA : IA[i] contains the index in A for the first non-zero element
        of row[i]. Thus IA[i+1] - IA[i] gives number of non-zero
        elements row[i]. The length of IA is always 1 more than the
        number of rows in the matrix.

    Examples
    ========

    >>> from sympy.matrices.sparsetools import _doktocsr
    >>> from sympy import SparseMatrix, diag
    >>> m = SparseMatrix(diag(1, 2, 3))
    >>> m[2, 0] = -1
    >>> _doktocsr(m)
    [[1, 2, -1, 3], [0, 1, 0, 2], [0, 1, 2, 4], [3, 3]]

    """
    row, JA, A = [list(i) for i in zip(*dok.row_list())]
    IA = [0] * ((row[0] if row else 0) + 1)
    for i, r in enumerate(row):
        IA.extend([i] * (r - row[i - 1]))
    IA.extend([len(A)] * (dok.rows - len(IA) + 1))
    shape = [dok.rows, dok.cols]
    return [A, JA, IA, shape]