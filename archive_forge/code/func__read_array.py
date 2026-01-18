import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_array(f, typecode, array_desc):
    """
    Read an array of type `typecode`, with the array descriptor given as
    `array_desc`.
    """
    if typecode in [1, 3, 4, 5, 6, 9, 13, 14, 15]:
        if typecode == 1:
            nbytes = _read_int32(f)
            if nbytes != array_desc['nbytes']:
                warnings.warn('Not able to verify number of bytes from header', stacklevel=3)
        array = np.frombuffer(f.read(array_desc['nbytes']), dtype=DTYPE_DICT[typecode])
    elif typecode in [2, 12]:
        array = np.frombuffer(f.read(array_desc['nbytes'] * 2), dtype=DTYPE_DICT[typecode])[1::2]
    else:
        array = []
        for i in range(array_desc['nelements']):
            dtype = typecode
            data = _read_data(f, dtype)
            array.append(data)
        array = np.array(array, dtype=np.object_)
    if array_desc['ndims'] > 1:
        dims = array_desc['dims'][:int(array_desc['ndims'])]
        dims.reverse()
        array = array.reshape(dims)
    _align_32(f)
    return array