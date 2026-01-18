from ml_dtypes._custom_floats import int4
from ml_dtypes._custom_floats import uint4
import numpy as np
class iinfo:
    kind: str
    bits: int
    min: int
    max: int
    dtype: np.dtype

    def __init__(self, int_type):
        if int_type == _int4_dtype:
            self.dtype = _int4_dtype
            self.kind = 'i'
            self.bits = 4
            self.min = -8
            self.max = 7
        elif int_type == _uint4_dtype:
            self.dtype = _uint4_dtype
            self.kind = 'u'
            self.bits = 4
            self.min = 0
            self.max = 15
        else:
            ii = np.iinfo(int_type)
            self.dtype = ii.dtype
            self.kind = ii.kind
            self.bits = ii.bits
            self.min = ii.min
            self.max = ii.max

    def __repr__(self):
        return f'iinfo(min={self.min}, max={self.max}, dtype={self.dtype})'

    def __str__(self):
        return repr(self)