import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
def _parse_nested_sequence(context, typ):
    """
    Parse a (possibly 0d) nested sequence type.
    A (ndim, dtype) tuple is returned.  Note the sequence may still be
    heterogeneous, as long as it converts to the given dtype.
    """
    if isinstance(typ, (types.Buffer,)):
        raise TypingError('%s not allowed in a homogeneous sequence' % typ)
    elif isinstance(typ, (types.Sequence,)):
        n, dtype = _parse_nested_sequence(context, typ.dtype)
        return (n + 1, dtype)
    elif isinstance(typ, (types.BaseTuple,)):
        if typ.count == 0:
            return (1, types.float64)
        n, dtype = _parse_nested_sequence(context, typ[0])
        dtypes = [dtype]
        for i in range(1, typ.count):
            _n, dtype = _parse_nested_sequence(context, typ[i])
            if _n != n:
                raise TypingError('type %s does not have a regular shape' % (typ,))
            dtypes.append(dtype)
        dtype = context.unify_types(*dtypes)
        if dtype is None:
            raise TypingError('cannot convert %s to a homogeneous type' % typ)
        return (n + 1, dtype)
    else:
        as_dtype(typ)
        return (0, typ)