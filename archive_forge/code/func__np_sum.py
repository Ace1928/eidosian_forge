def _np_sum(a, axis=None, dtype=None, keepdims=False, initial=None, out=None):
    """
    Sum of array elements over a given axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int, optional
        Axis or axes along which a sum is performed.  The default,
        axis=None, will sum all of the elements of the input array.  If
        axis is negative it counts from the last to the first axis.
    dtype : dtype, optional
        The type of the returned array and of the accumulator in which the
        elements are summed. The default type is float32.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.

        If the default value is passed, then `keepdims` will not be
        passed through to the `sum` method of sub-classes of
        `ndarray`, however any non-default value will be.  If the
        sub-classes `sum` method does not implement `keepdims` any
        exceptions will be raised.
    initial: Currently only supports None as input, optional
        Starting value for the sum.
        Currently not implemented. Please use ``None`` as input or skip this argument.
    out : ndarray or None, optional
        Alternative output array in which to place the result. It must have
        the same shape and dtype as the expected output.

    Returns
    -------
    sum_along_axis : ndarray
        An ndarray with the same shape as `a`, with the specified
        axis removed. If an output array is specified, a reference to
        `out` is returned.

    Notes
    -----
    - Input type does not support Python native iterables.
    - "out" param: cannot perform auto type change. out ndarray's dtype must be the same as the expected output.
    - "initial" param is not supported yet. Please use None as input.
    - Arithmetic is modular when using integer types, and no error is raised on overflow.
    - The sum of an empty array is the neutral element 0:

    >>> a = np.empty(1)
    >>> np.sum(a)
    array(0.)

    This function differs from the original `numpy.sum
    <https://docs.scipy.org/doc/numpy/reference/generated/numpy.sum.html>`_ in
    the following aspects:

    - Input type does not support Python native iterables(list, tuple, ...).
    - "out" param: cannot perform auto type cast. out ndarray's dtype must be the same as the expected output.
    - "initial" param is not supported yet. Please use ``None`` as input or skip it.

    Examples
    --------
    >>> a = np.array([0.5, 1.5])
    >>> np.sum(a)
    array(2.)
    >>> a = np.array([0.5, 0.7, 0.2, 1.5])
    >>> np.sum(a, dtype=np.int32)
    array(2, dtype=int32)
    >>> a = np.array([[0, 1], [0, 5]])
    >>> np.sum(a)
    array(6.)
    >>> np.sum(a, axis=0)
    array([0., 6.])
    >>> np.sum(a, axis=1)
    array([1., 5.])

    With output ndarray:

    >>> a = np.array([[0, 1], [0, 5]])
    >>> b = np.ones((2,), dtype=np.float32)
    >>> np.sum(a, axis = 0, out=b)
    array([0., 6.])
    >>> b
    array([0., 6.])

    If the accumulator is too small, overflow occurs:

    >>> np.ones(128, dtype=np.int8).sum(dtype=np.int8)
    array(-128, dtype=int8)
    """
    pass