import numpy as np
def _kind_name(dtype):
    try:
        return _kind_to_stem[dtype.kind]
    except KeyError as e:
        raise RuntimeError('internal dtype error, unknown kind {!r}'.format(dtype.kind)) from None