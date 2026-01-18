import numpy as np
def _scalar_str(dtype, short):
    byteorder = _byte_order_str(dtype)
    if dtype.type == np.bool_:
        if short:
            return "'?'"
        else:
            return "'bool'"
    elif dtype.type == np.object_:
        return "'O'"
    elif dtype.type == np.bytes_:
        if _isunsized(dtype):
            return "'S'"
        else:
            return "'S%d'" % dtype.itemsize
    elif dtype.type == np.str_:
        if _isunsized(dtype):
            return "'%sU'" % byteorder
        else:
            return "'%sU%d'" % (byteorder, dtype.itemsize / 4)
    elif issubclass(dtype.type, np.void):
        if _isunsized(dtype):
            return "'V'"
        else:
            return "'V%d'" % dtype.itemsize
    elif dtype.type == np.datetime64:
        return "'%sM8%s'" % (byteorder, _datetime_metadata_str(dtype))
    elif dtype.type == np.timedelta64:
        return "'%sm8%s'" % (byteorder, _datetime_metadata_str(dtype))
    elif np.issubdtype(dtype, np.number):
        if short or dtype.byteorder not in ('=', '|'):
            return "'%s%c%d'" % (byteorder, dtype.kind, dtype.itemsize)
        else:
            return "'%s%d'" % (_kind_name(dtype), 8 * dtype.itemsize)
    elif dtype.isbuiltin == 2:
        return dtype.type.__name__
    else:
        raise RuntimeError('Internal error: NumPy dtype unrecognized type number')