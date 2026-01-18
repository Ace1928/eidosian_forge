import warnings
from warnings import warn
import numpy as np
def _dtype_bits(kind, bits, itemsize=1):
    """Return dtype of `kind` that can store a `bits` wide unsigned int

    Parameters:
    kind: str
        Data type kind.
    bits: int
        Desired number of bits.
    itemsize: int
        The data type object element size.

    Returns
    -------
    dtype: data type object
        Data type of `kind` that can store a `bits` wide unsigned int

    """
    s = next((i for i in (itemsize,) + (2, 4, 8) if bits < i * 8 or (bits == i * 8 and kind == 'u')))
    return np.dtype(kind + str(s))