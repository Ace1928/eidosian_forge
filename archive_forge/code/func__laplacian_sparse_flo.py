import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
def _laplacian_sparse_flo(graph, normed, axis, copy, form, dtype, symmetrized):
    del copy
    if dtype is None:
        dtype = graph.dtype
    graph_sum = np.asarray(graph.sum(axis=axis)).ravel()
    graph_diagonal = graph.diagonal()
    diag = graph_sum - graph_diagonal
    if symmetrized:
        graph_sum += np.asarray(graph.sum(axis=1 - axis)).ravel()
        diag = graph_sum - graph_diagonal - graph_diagonal
    if normed:
        isolated_node_mask = diag == 0
        w = np.where(isolated_node_mask, 1, np.sqrt(diag))
        if symmetrized:
            md = _laplace_normed_sym(graph, graph_sum, 1.0 / w)
        else:
            md = _laplace_normed(graph, graph_sum, 1.0 / w)
        if form == 'function':
            return (md, w.astype(dtype, copy=False))
        elif form == 'lo':
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return (m, w.astype(dtype, copy=False))
        else:
            raise ValueError(f'Invalid form: {form!r}')
    else:
        if symmetrized:
            md = _laplace_sym(graph, graph_sum)
        else:
            md = _laplace(graph, graph_sum)
        if form == 'function':
            return (md, diag.astype(dtype, copy=False))
        elif form == 'lo':
            m = _linearoperator(md, shape=graph.shape, dtype=dtype)
            return (m, diag.astype(dtype, copy=False))
        else:
            raise ValueError(f'Invalid form: {form!r}')