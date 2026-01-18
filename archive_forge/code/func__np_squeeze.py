def _np_squeeze(a, axis=None, out=None):
    """
    Remove single-dimensional entries from the shape of an array.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int or tuple of ints, optional
        Selects a subset of the single-dimensional entries in the
        shape. If an axis is selected with shape entry greater than
        one, an error is raised.
    out : ndarray, optional
        Array into which the output is placed. It must have the same size
        and dtype as the input array.

    Returns
    -------
    squeezed : ndarray
        The input array, but with all or a subset of the
        dimensions of length 1 removed. It always returns a copy of `a`.

    Raises
    ------
    MXNetError
        If `axis` is not `None`, and an axis being squeezed is not of length 1

    See Also
    --------
    expand_dims : The inverse operation, adding singleton dimensions
    reshape : Insert, remove, and combine dimensions, and resize existing ones

    Examples
    --------
    >>> x = np.array([[[0], [1], [2]]])
    >>> x.shape
    (1, 3, 1)
    >>> np.squeeze(x).shape
    (3,)
    >>> np.squeeze(x, axis=0).shape
    (3, 1)
    >>> np.squeeze(x, axis=1).shape
    Traceback (most recent call last):
    ...
    mxnet.base.MXNetError: cannot select an axis to squeeze out which has size=3 not equal to one
    >>> np.squeeze(x, axis=2).shape
    (1, 3)
    """
    pass