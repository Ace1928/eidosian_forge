import numpy as np
import scipy
import scipy.sparse.linalg
import scipy.stats
import threadpoolctl
import sklearn
from ..externals._packaging.version import parse as parse_version
from .deprecation import deprecated
def _preserve_dia_indices_dtype(sparse_container, original_container_format, requested_sparse_format):
    """Preserve indices dtype for SciPy < 1.12 when converting from DIA to CSR/CSC.

    For SciPy < 1.12, DIA arrays indices are upcasted to `np.int64` that is
    inconsistent with DIA matrices. We downcast the indices dtype to `np.int32` to
    be consistent with DIA matrices.

    The converted indices arrays are affected back inplace to the sparse container.

    Parameters
    ----------
    sparse_container : sparse container
        Sparse container to be checked.
    requested_sparse_format : str or bool
        The type of format of `sparse_container`.

    Notes
    -----
    See https://github.com/scipy/scipy/issues/19245 for more details.
    """
    if original_container_format == 'dia_array' and requested_sparse_format in ('csr', 'coo'):
        if requested_sparse_format == 'csr':
            index_dtype = _smallest_admissible_index_dtype(arrays=(sparse_container.indptr, sparse_container.indices), maxval=max(sparse_container.nnz, sparse_container.shape[1]), check_contents=True)
            sparse_container.indices = sparse_container.indices.astype(index_dtype, copy=False)
            sparse_container.indptr = sparse_container.indptr.astype(index_dtype, copy=False)
        else:
            index_dtype = _smallest_admissible_index_dtype(maxval=max(sparse_container.shape))
            sparse_container.row = sparse_container.row.astype(index_dtype, copy=False)
            sparse_container.col = sparse_container.col.astype(index_dtype, copy=False)