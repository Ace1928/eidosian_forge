from cupy import _core
import cupy
def _create_float_test_ufunc(name, doc):
    return _core.create_ufunc('cupy_' + name, ('e->?', 'f->?', 'd->?', 'F->?', 'D->?'), 'out0 = %s(in0)' % name, doc=doc)