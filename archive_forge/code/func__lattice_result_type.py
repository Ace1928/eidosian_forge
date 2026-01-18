import functools
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common.variables import ALLOWED_DTYPES
from keras.src.backend.common.variables import standardize_dtype
def _lattice_result_type(*args):
    dtypes, weak_types = zip(*(_dtype_and_weaktype(arg) for arg in args))
    if len(dtypes) == 1:
        out_dtype = dtypes[0]
        out_weak_type = weak_types[0]
    elif len(set(dtypes)) == 1 and (not all(weak_types)):
        out_dtype = dtypes[0]
        out_weak_type = False
    elif all(weak_types):
        out_dtype = _least_upper_bound(*{_respect_weak_type(d, False) for d in dtypes})
        out_weak_type = True
    else:
        out_dtype = _least_upper_bound(*{_respect_weak_type(d, w) for d, w in zip(dtypes, weak_types)})
        out_weak_type = any((out_dtype is t for t in WEAK_TYPES))
    out_weak_type = out_dtype != 'bool' and out_weak_type
    precision = backend.floatx()[-2:]
    if out_weak_type:
        out_dtype = _resolve_weak_type(out_dtype, precision=precision)
    return out_dtype