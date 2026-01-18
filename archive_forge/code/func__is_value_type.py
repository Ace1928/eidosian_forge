from __future__ import absolute_import
import math, sys
def _is_value_type(t):
    if isinstance(t, typedef):
        return _is_value_type(t._basetype)
    return isinstance(t, type) and issubclass(t, (StructType, UnionType, ArrayType))