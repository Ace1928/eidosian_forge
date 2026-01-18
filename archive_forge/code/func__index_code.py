from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def _index_code(idx):
    if idx.is_slice:
        values = (idx.start, idx.stop, idx.step)
        if idx.step.is_none:
            func = 'contiguous_slice'
            values = values[:2]
        else:
            func = 'slice'
        return 'pythonic::types::%s(%s)' % (func, ','.join((v.pythran_result() for v in values)))
    elif idx.type.is_int:
        return to_pythran(idx)
    elif idx.type.is_pythran_expr:
        return idx.pythran_result()
    raise ValueError('unsupported indexing type %s' % idx.type)