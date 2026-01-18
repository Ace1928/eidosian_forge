import asyncio
import builtins
import functools
import inspect
from typing import Callable, Optional
import numpy as np
from numpy.lib.function_base import (
def _update_arrays_type(arrays, results):
    """Update the dtype of arrays.

    String arrays contain strings of fixed length. Here they are initialized with
    the type of the first results, so that if the next results contain longer
    strings they will be truncated when added to the output arrays. Here we
    update the type if the current results contain longer strings than in the
    current output array.

    Parameters
    ----------
    arrays
        Arrays that contain the vectorized function's results.
    results
        The current output of the function being vectorized.

    """
    updated_arrays = []
    for array, result in zip(arrays, results):
        if array.dtype.type == np.str_:
            if array.dtype < np.array(result).dtype:
                array = array.astype(np.array(result).dtype)
        updated_arrays.append(array)
    return tuple(updated_arrays)