from sys import version_info as _swig_python_version_info
class ProblemData(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc='The membership flag')
    __repr__ = _swig_repr
    TensorV = property(_cvxcore.ProblemData_TensorV_get, _cvxcore.ProblemData_TensorV_set)
    TensorI = property(_cvxcore.ProblemData_TensorI_get, _cvxcore.ProblemData_TensorI_set)
    TensorJ = property(_cvxcore.ProblemData_TensorJ_get, _cvxcore.ProblemData_TensorJ_set)
    param_id = property(_cvxcore.ProblemData_param_id_get, _cvxcore.ProblemData_param_id_set)
    vec_idx = property(_cvxcore.ProblemData_vec_idx_get, _cvxcore.ProblemData_vec_idx_set)

    def init_id(self, new_param_id, param_size):
        return _cvxcore.ProblemData_init_id(self, new_param_id, param_size)

    def getLen(self):
        return _cvxcore.ProblemData_getLen(self)

    def getV(self, values):
        return _cvxcore.ProblemData_getV(self, values)

    def getI(self, values):
        return _cvxcore.ProblemData_getI(self, values)

    def getJ(self, values):
        return _cvxcore.ProblemData_getJ(self, values)

    def __init__(self):
        _cvxcore.ProblemData_swiginit(self, _cvxcore.new_ProblemData())
    __swig_destroy__ = _cvxcore.delete_ProblemData