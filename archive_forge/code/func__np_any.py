def _np_any(a, axis=None, keepdims=False, out=None):
    """
    Test whether any array element along a given axis evaluates to True.
    Returns single boolean unless axis is not None 

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
    any : bool or ndarray
        A new boolean or ndarray is returned unless out is specified,
        in which case a reference to out is returned.

    Examples:
    ---------
    >>> np.any([[True, False], [True, True]])
    True

    >>> np.any([[True, False], [False, False]], axis=0)
    array([ True, False])

    >>> np.any([-1, 0, 5])
    True

    >>> np.any(np.nan)
    True

    >>> o=np.array(False)
    >>> z=np.any([-1, 4, 5], out=o)
    >>> z, o
    (array(True), array(True))
    >>> # Check now that z is a reference to o
    >>> z is o
    True
    >>> id(z), id(o) # identity of z and o              # doctest: +SKIP
    (191614240, 191614240)
    """
    pass