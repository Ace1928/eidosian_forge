def _np_amax(a, axis=None, keepdims=False, out=None):
    """
    Return the maximum of an array or maximum along an axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : int, optional
        Axis along which to operate.  By default, flattened input is used.
    out : ndarray, optional
        Alternative output array in which to place the result.  Must
        be of the same shape and buffer length as the expected output.
        See `doc.ufuncs` (Section "Output arguments") for more details.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    amax : ndarray
        Maximum of `a`. If `axis` is None, the result is an array of dimension 1.
        If `axis` is given, the result is an array of dimension
        ``a.ndim - 1``.

    See Also
    --------
    min :
        The minimum value of an array along a given axis, ignoring any nan.
    maximum :
        Element-wise maximum of two arrays, ignoring any nan.
    argmax :
        Return the indices of the maximum values.

    Notes
    -----
    NaN in the orginal `numpy` is denoted as nan and will be ignored.

    Don't use `amax` for element-wise comparison of 2 arrays; when
    ``a.shape[0]`` is 2, ``maximum(a[0], a[1])`` is faster than
    ``amax(a, axis=0)``.

    Examples
    --------
    >>> a = np.arange(4).reshape((2,2))
    >>> a
    array([[0., 1.],
        [2., 3.]])
    >>> np.amax(a)            # Maximum of the flattened array
    array(3.)
    >>> np.amax(a, axis=0)    # Maxima along the first axis
    array([2., 3.])
    >>> np.amax(a, axis=1)    # Maxima along the second axis
    array([1., 3.])

    >>> b = np.arange(5, dtype=np.float32)
    >>> b[2] = np.nan
    >>> np.amax(b)
    array(4.)
    """
    pass