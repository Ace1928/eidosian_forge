from math import sqrt
from numbers import Real
import numpy as np
from scipy import ndimage as ndi
Normalize spacing parameter.

    The `spacing` parameter should be a sequence of numbers matching
    the image dimensions. If `spacing` is a scalar, assume equal
    spacing along all dimensions.

    Parameters
    ---------
    spacing : Any
        User-provided `spacing` keyword.
    ndims : int
        Number of image dimensions.

    Returns
    -------
    spacing : array
        Corrected spacing.

    Raises
    ------
    ValueError
        If `spacing` is invalid.

    