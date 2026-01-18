import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
def _laplacian_dense_flo(graph, normed, axis, copy, form, dtype, symmetrized):
    if copy:
        m = np.array(graph)
    else:
        m = np.asarray(graph)
    if dtype is None:
        dtype = m.dtype
    graph_sum = m.sum(axis=axis)
    graph_diagonal = m.diagonal()
    diag = graph_sum - graph_diagonal
    if symmetrized:
        graph_sum += m.sum(axis=1 - axis)
        diag = graph_sum - graph_diagonal - graph_diagonal
    if normed:
        isolated_node_mask = diag == 0
        w = np.where(isolated_node_mask, 1, np.sqrt(diag))
        if symmetrized:
            md = _laplace_normed_sym(m, graph_sum, 1.0 / w)
        else:
            md = _laplace_normed(m, graph_sum, 1.0 / w)
        if form == 'function':
            return (md, w.astype(dtype, copy=False))
        elif form == 'lo':
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return (m, w.astype(dtype, copy=False))
        else:
            raise ValueError(f'Invalid form: {form!r}')
    else:
        if symmetrized:
            md = _laplace_sym(m, graph_sum)
        else:
            md = _laplace(m, graph_sum)
        if form == 'function':
            return (md, diag.astype(dtype, copy=False))
        elif form == 'lo':
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return (m, diag.astype(dtype, copy=False))
        else:
            raise ValueError(f'Invalid form: {form!r}')