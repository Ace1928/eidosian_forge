def _np_all(a, axis=None, keepdims=False, out=None):
    """
    Test whether all array elements along a given axis evaluate to True.

    Parameters
    ----------
    a : array_like
        Input array or object that can be converted to an array.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a logical AND reduction is performed.
        The default (axis = None) is to perform a logical AND over
        all the dimensions of the input array.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left in
        the result as dimensions with size one. With this option,
        the result will broadcast correctly against the input array.
    out : ndarray, optional
        Alternate output array in which to place the result. It must have
        the same shape as the expected output and its type is preserved

    Returns
    --------
    all : ndarray, bool
        A new boolean or array is returned unless out is specified,
        in which case a reference to out is returned.

    Examples:
    ---------
    >>> np.all([[True,False],[True,True]])
    False

    >>> np.all([[True,False],[True,True]], axis=0)
    array([ True, False])

    >>> np.all([-1, 4, 5])
    True

    >>> np.all([1.0, np.nan])
    True

    >>> o=np.array(False)
    >>> z=np.all([-1, 4, 5], out=o)
    >>> id(z), id(o), z
    (28293632, 28293632, array(True)) # may vary
    """
    pass