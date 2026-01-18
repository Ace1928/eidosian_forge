import functools
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.backend.common.variables import ALLOWED_DTYPES
from keras.src.backend.common.variables import standardize_dtype
def _type_promotion_lattice():
    """
    Return the type promotion lattice in the form of a DAG.
    This DAG maps each type to its immediately higher type on the lattice.
    """
    b1, = BOOL_TYPES
    u1, u2, u4, u8, i1, i2, i4, i8 = INT_TYPES
    bf, f2, f4, f8 = FLOAT_TYPES
    i_, f_ = WEAK_TYPES
    out = {b1: [i_], u1: [i2, u2], u2: [i4, u4], u4: [i8, u8], u8: [f_], i_: [u1, i1], i1: [i2], i2: [i4], i4: [i8], i8: [f_], f_: [bf, f2], bf: [f4], f2: [f4], f4: [f8], f8: []}
    return out