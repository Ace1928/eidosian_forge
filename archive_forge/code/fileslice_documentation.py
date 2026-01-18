import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
Return array shape `shape` where all entries point to value `scalar`

    Parameters
    ----------
    shape : sequence
        Shape of output array.
    scalar : scalar
        Scalar value with which to fill array.

    Returns
    -------
    strided_arr : array
        Array of shape `shape` for which all values == `scalar`, built by
        setting all strides of `strided_arr` to 0, so the scalar is broadcast
        out to the full array `shape`. `strided_arr` is flagged as not
        `writeable`.

        The array is set read-only to avoid a numpy error when broadcasting -
        see https://github.com/numpy/numpy/issues/6491
    