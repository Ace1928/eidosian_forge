import itertools
import math
import numpy as np
Analytical formulae for 2D and 3D central moments of order < 4.

    `moments_raw_to_central` will automatically call this function when
    ndim < 4 and order < 4.

    Parameters
    ----------
    moments_raw : ndarray
        The raw moments.

    Returns
    -------
    moments_central : ndarray
        The central moments.
    