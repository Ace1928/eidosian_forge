import pytest
import numpy as np
from numpy.testing import assert_allclose
from pytest import raises as assert_raises
from scipy import sparse
from scipy.sparse import csgraph
from scipy._lib._util import np_long, np_ulong
def _check_symmetric_graph_laplacian(mat, normed, copy=True):
    if not hasattr(mat, 'shape'):
        mat = eval(mat, dict(np=np, sparse=sparse))
    if sparse.issparse(mat):
        sp_mat = mat
        mat = sp_mat.toarray()
    else:
        sp_mat = sparse.csr_matrix(mat)
    mat_copy = np.copy(mat)
    sp_mat_copy = sparse.csr_matrix(sp_mat, copy=True)
    n_nodes = mat.shape[0]
    explicit_laplacian = _explicit_laplacian(mat, normed=normed)
    laplacian = csgraph.laplacian(mat, normed=normed, copy=copy)
    sp_laplacian = csgraph.laplacian(sp_mat, normed=normed, copy=copy)
    if copy:
        assert_allclose(mat, mat_copy)
        _assert_allclose_sparse(sp_mat, sp_mat_copy)
    elif not (normed and check_int_type(mat)):
        assert_allclose(laplacian, mat)
        if sp_mat.format == 'coo':
            _assert_allclose_sparse(sp_laplacian, sp_mat)
    assert_allclose(laplacian, sp_laplacian.toarray())
    for tested in [laplacian, sp_laplacian.toarray()]:
        if not normed:
            assert_allclose(tested.sum(axis=0), np.zeros(n_nodes))
        assert_allclose(tested.T, tested)
        assert_allclose(tested, explicit_laplacian)