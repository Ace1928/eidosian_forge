def _np_repeat(a, repeats, axis=None):
    """
    Repeat elements of an array.

    Parameters
    ----------
    a : ndarray
        Input array.
    repeats : int
        The number of repetitions for each element.
    axis : int, optional
        The axis along which to repeat values.  By default, use the
        flattened input array, and return a flat output array.

    Returns
    -------
    repeated_array : ndarray
        Output array which has the same shape as `a`, except along
        the given axis.

    Notes
    -----
    Unlike the official NumPy ``repeat`` operator, this operator currently
    does not support array of ints for the parameter `repeats`.

    Examples
    --------
    >>> x = np.arange(4).reshape(2, 2)
    >>> x
    array([[0., 1.],
           [2., 3.]])
    >>> np.repeat(x, repeats=3)
    array([0., 0., 0., 1., 1., 1., 2., 2., 2., 3., 3., 3.])
    >>> np.repeat(x, repeats=3, axis=0)
    array([[0., 1.],
           [0., 1.],
           [0., 1.],
           [2., 3.],
           [2., 3.],
           [2., 3.]])
    >>> np.repeat(x, repeats=3, axis=1)
    array([[0., 0., 0., 1., 1., 1.],
           [2., 2., 2., 3., 3., 3.]])
    """
    pass