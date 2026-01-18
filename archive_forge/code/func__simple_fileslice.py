import operator
from functools import reduce
from mmap import mmap
from numbers import Integral
import numpy as np
def _simple_fileslice(fileobj, sliceobj, shape, dtype, offset=0, order='C', heuristic=None):
    """Read all data from `fileobj` into array, then slice with `sliceobj`

    The simplest possible thing; read all the data into the full array, then
    slice the full array.

    Parameters
    ----------
    fileobj : file-like object
        implements ``read`` and ``seek``
    sliceobj : object
        something that can be used to slice an array as in ``arr[sliceobj]``
    shape : sequence
        shape of full array inside `fileobj`
    dtype : dtype object
        dtype of array inside `fileobj`
    offset : int, optional
        offset of array data within `fileobj`
    order : {'C', 'F'}, optional
        memory layout of array in `fileobj`
    heuristic : optional
        The routine doesn't use `heuristic`; the parameter is for API
        compatibility with :func:`fileslice`

    Returns
    -------
    sliced_arr : array
        Array in `fileobj` as sliced with `sliceobj`
    """
    fileobj.seek(offset)
    nbytes = reduce(operator.mul, shape) * dtype.itemsize
    bytes = fileobj.read(nbytes)
    new_arr = np.ndarray(shape, dtype, buffer=bytes, order=order)
    return new_arr[sliceobj]