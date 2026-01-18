import functools
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common.variables import ALLOWED_DTYPES
from keras.src.backend.common.variables import standardize_dtype
@functools.lru_cache(maxsize=None)
def _respect_weak_type(dtype, weak_type):
    """Return the weak dtype of `dtype` if `weak_type==True`."""
    if weak_type:
        if dtype == 'bool':
            return dtype
        elif 'float' in dtype:
            return 'float'
        elif 'int' in dtype:
            return 'int'
        else:
            raise ValueError(f'Invalid value for argument `dtype`. Expected one of {ALLOWED_DTYPES}. Received: dtype={dtype}')
    return dtype