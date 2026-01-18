import struct
import numpy as np
import tempfile
import zlib
import warnings
def _read_structure(f, array_desc, struct_desc):
    """
    Read a structure, with the array and structure descriptors given as
    `array_desc` and `structure_desc` respectively.
    """
    nrows = array_desc['nelements']
    columns = struct_desc['tagtable']
    dtype = []
    for col in columns:
        if col['structure'] or col['array']:
            dtype.append(((col['name'].lower(), col['name']), np.object_))
        elif col['typecode'] in DTYPE_DICT:
            dtype.append(((col['name'].lower(), col['name']), DTYPE_DICT[col['typecode']]))
        else:
            raise Exception('Variable type %i not implemented' % col['typecode'])
    structure = np.rec.recarray((nrows,), dtype=dtype)
    for i in range(nrows):
        for col in columns:
            dtype = col['typecode']
            if col['structure']:
                structure[col['name']][i] = _read_structure(f, struct_desc['arrtable'][col['name']], struct_desc['structtable'][col['name']])
            elif col['array']:
                structure[col['name']][i] = _read_array(f, dtype, struct_desc['arrtable'][col['name']])
            else:
                structure[col['name']][i] = _read_data(f, dtype)
    if array_desc['ndims'] > 1:
        dims = array_desc['dims'][:int(array_desc['ndims'])]
        dims.reverse()
        structure = structure.reshape(dims)
    return structure