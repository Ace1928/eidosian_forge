import os
from numpy import (
from numpy.core.multiarray import _flagdict, flagsobj
def _ctype_from_dtype_structured(dtype):
    field_data = []
    for name in dtype.names:
        field_dtype, offset = dtype.fields[name][:2]
        field_data.append((offset, name, _ctype_from_dtype(field_dtype)))
    field_data = sorted(field_data, key=lambda f: f[0])
    if len(field_data) > 1 and all((offset == 0 for offset, name, ctype in field_data)):
        size = 0
        _fields_ = []
        for offset, name, ctype in field_data:
            _fields_.append((name, ctype))
            size = max(size, ctypes.sizeof(ctype))
        if dtype.itemsize != size:
            _fields_.append(('', ctypes.c_char * dtype.itemsize))
        return type('union', (ctypes.Union,), dict(_fields_=_fields_, _pack_=1, __module__=None))
    else:
        last_offset = 0
        _fields_ = []
        for offset, name, ctype in field_data:
            padding = offset - last_offset
            if padding < 0:
                raise NotImplementedError('Overlapping fields')
            if padding > 0:
                _fields_.append(('', ctypes.c_char * padding))
            _fields_.append((name, ctype))
            last_offset = offset + ctypes.sizeof(ctype)
        padding = dtype.itemsize - last_offset
        if padding > 0:
            _fields_.append(('', ctypes.c_char * padding))
        return type('struct', (ctypes.Structure,), dict(_fields_=_fields_, _pack_=1, __module__=None))