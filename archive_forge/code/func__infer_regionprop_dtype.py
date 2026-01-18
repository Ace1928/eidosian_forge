import inspect
from functools import wraps
from math import atan2
from math import pi as PI
from math import sqrt
from warnings import warn
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.distance import pdist
from . import _moments
from ._find_contours import find_contours
from ._marching_cubes_lewiner import marching_cubes
from ._regionprops_utils import (
def _infer_regionprop_dtype(func, *, intensity, ndim):
    """Infer the dtype of a region property calculated by func.

    If a region property function always returns the same shape and type of
    output regardless of input size, then the dtype is the dtype of the
    returned array. Otherwise, the property has object dtype.

    Parameters
    ----------
    func : callable
        Function to be tested. The signature should be array[bool] -> Any if
        intensity is False, or *(array[bool], array[float]) -> Any otherwise.
    intensity : bool
        Whether the regionprop is calculated on an intensity image.
    ndim : int
        The number of dimensions for which to check func.

    Returns
    -------
    dtype : NumPy data type
        The data type of the returned property.
    """
    mask_1 = np.ones((1,) * ndim, dtype=bool)
    mask_1 = np.pad(mask_1, (0, 1), constant_values=False)
    mask_2 = np.ones((2,) * ndim, dtype=bool)
    mask_2 = np.pad(mask_2, (1, 0), constant_values=False)
    propmasks = [mask_1, mask_2]
    rng = np.random.default_rng()
    if intensity and _infer_number_of_required_args(func) == 2:

        def _func(mask):
            return func(mask, rng.random(mask.shape))
    else:
        _func = func
    props1, props2 = map(_func, propmasks)
    if np.isscalar(props1) and np.isscalar(props2) or np.array(props1).shape == np.array(props2).shape:
        dtype = np.array(props1).dtype.type
    else:
        dtype = np.object_
    return dtype