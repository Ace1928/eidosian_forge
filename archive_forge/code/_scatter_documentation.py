Stores a minimum value of elements specified by indices to an array.

    It stores the minimum value of elements in ``value`` array indexed by
    ``slices`` to ``a``. If all of the indices target different locations,
    the operation of :func:`scatter_min` is equivalent to
    ``a[slices] = cupy.minimum(a[slices], value)``.
    If there are multiple elements targeting the same location,
    :func:`scatter_min` stores the minimum of all of these values to the given
    index of ``a``, the initial element of ``a`` is also taken in account.

    Note that just like an array indexing, negative indices are interpreted as
    counting from the end of an array.

    Also note that :func:`scatter_min` behaves identically
    to :func:`numpy.minimum.at`.

    Example
    -------
    >>> import numpy
    >>> import cupy
    >>> a = cupy.zeros((6,), dtype=numpy.float32)
    >>> i = cupy.array([1, 0, 1, 2])
    >>> v = cupy.array([1., 2., 3., -1.])
    >>> cupyx.scatter_min(a, i, v);
    >>> a
    array([ 0.,  0., -1.,  0.,  0.,  0.], dtype=float32)

    Args:
        a (ndarray): An array to store the results.
        slices: It is integer, slices, ellipsis, numpy.newaxis,
            integer array-like, boolean array-like or tuple of them.
            It works for slices used for
            :func:`cupy.ndarray.__getitem__` and
            :func:`cupy.ndarray.__setitem__`.
        v (array-like): An array used for reference.
    