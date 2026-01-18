import numpy as np
def _name_get(dtype):
    if dtype.isbuiltin == 2:
        return dtype.type.__name__
    if dtype.kind == '\x00':
        name = type(dtype).__name__
    elif issubclass(dtype.type, np.void):
        name = dtype.type.__name__
    else:
        name = _kind_name(dtype)
    if _name_includes_bit_suffix(dtype):
        name += '{}'.format(dtype.itemsize * 8)
    if dtype.type in (np.datetime64, np.timedelta64):
        name += _datetime_metadata_str(dtype)
    return name