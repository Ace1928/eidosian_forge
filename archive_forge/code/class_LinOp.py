from sys import version_info as _swig_python_version_info
class LinOp(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr

    def __init__(self, type, shape, args):
        _cvxcore.LinOp_swiginit(self, _cvxcore.new_LinOp(type, shape, args))

    def get_type(self):
        return _cvxcore.LinOp_get_type(self)

    def is_constant(self):
        return _cvxcore.LinOp_is_constant(self)

    def get_shape(self):
        return _cvxcore.LinOp_get_shape(self)

    def get_args(self):
        return _cvxcore.LinOp_get_args(self)

    def get_slice(self):
        return _cvxcore.LinOp_get_slice(self)

    def push_back_slice_vec(self, slice_vec):
        return _cvxcore.LinOp_push_back_slice_vec(self, slice_vec)

    def has_numerical_data(self):
        return _cvxcore.LinOp_has_numerical_data(self)

    def get_linOp_data(self):
        return _cvxcore.LinOp_get_linOp_data(self)

    def set_linOp_data(self, tree):
        return _cvxcore.LinOp_set_linOp_data(self, tree)

    def get_data_ndim(self):
        return _cvxcore.LinOp_get_data_ndim(self)

    def set_data_ndim(self, ndim):
        return _cvxcore.LinOp_set_data_ndim(self, ndim)

    def is_sparse(self):
        return _cvxcore.LinOp_is_sparse(self)

    def get_sparse_data(self):
        return _cvxcore.LinOp_get_sparse_data(self)

    def get_dense_data(self):
        return _cvxcore.LinOp_get_dense_data(self)

    def set_dense_data(self, matrix):
        return _cvxcore.LinOp_set_dense_data(self, matrix)

    def set_sparse_data(self, data, row_idxs, col_idxs, rows, cols):
        return _cvxcore.LinOp_set_sparse_data(self, data, row_idxs, col_idxs, rows, cols)
    __swig_destroy__ = _cvxcore.delete_LinOp