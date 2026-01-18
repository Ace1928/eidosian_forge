import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
@infer
class StaticGetItemArray(AbstractTemplate):
    key = 'static_getitem'

    def generic(self, args, kws):
        ary, idx = args
        if isinstance(ary, types.Array) and isinstance(idx, str) and isinstance(ary.dtype, types.Record):
            if idx in ary.dtype.fields:
                attr_dtype = ary.dtype.typeof(idx)
                if isinstance(attr_dtype, types.NestedArray):
                    ret = ary.copy(dtype=attr_dtype.dtype, ndim=ary.ndim + attr_dtype.ndim, layout='A')
                    return signature(ret, *args)
                else:
                    ret = ary.copy(dtype=attr_dtype, layout='A')
                    return signature(ret, *args)