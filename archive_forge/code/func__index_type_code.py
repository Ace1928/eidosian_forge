from __future__ import absolute_import
from .PyrexTypes import CType, CTypedefType, CStructOrUnionType
import cython
def _index_type_code(index_with_type):
    idx, index_type = index_with_type
    if idx.is_slice:
        n = 2 + int(not idx.step.is_none)
        return 'pythonic::%s::functor::slice{}(%s)' % (pythran_builtins, ','.join(['0'] * n))
    elif index_type.is_int:
        return 'std::declval<%s>()' % index_type.sign_and_name()
    elif index_type.is_pythran_expr:
        return 'std::declval<%s>()' % index_type.pythran_type
    raise ValueError('unsupported indexing type %s!' % index_type)