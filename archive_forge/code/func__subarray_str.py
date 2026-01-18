import numpy as np
def _subarray_str(dtype):
    base, shape = dtype.subdtype
    return '({}, {})'.format(_construction_repr(base, short=True), shape)