import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
def _laplacian_dense(graph, normed, axis, copy, form, dtype, symmetrized):
    if form != 'array':
        raise ValueError(f'{form!r} must be "array"')
    if dtype is None:
        dtype = graph.dtype
    if copy:
        m = np.array(graph)
    else:
        m = np.asarray(graph)
    if dtype is None:
        dtype = m.dtype
    if symmetrized:
        m += m.T.conj()
    np.fill_diagonal(m, 0)
    w = m.sum(axis=axis)
    if normed:
        isolated_node_mask = w == 0
        w = np.where(isolated_node_mask, 1, np.sqrt(w))
        m /= w
        m /= w[:, np.newaxis]
        m *= -1
        _setdiag_dense(m, 1 - isolated_node_mask)
    else:
        m *= -1
        _setdiag_dense(m, w)
    return (m.astype(dtype, copy=False), w.astype(dtype, copy=False))