def _np_moveaxis(a, source, destination):
    """Move axes of an array to new positions.
    Other axes remain in their original order.

    Parameters
    ----------
    a : ndarray
        The array whose axes should be reordered.
        source : int or sequence of int
        Original positions of the axes to move. These must be unique.
        destination : int or sequence of int
        Destination positions for each of the original axes. These must also be
        unique.

    Returns
    -------
    result : ndarray
        Array with moved axes. This array is a view of the input array.

    See Also
    --------
        transpose: Permute the dimensions of an array.
        swapaxes: Interchange two axes of an array.

    Examples
    --------
    >>> x = np.zeros((3, 4, 5))
    >>> np.moveaxis(x, 0, -1).shape
    (4, 5, 3)
    >>> np.moveaxis(x, -1, 0).shape
    (5, 3, 4)
    These all achieve the same result:
    >>> np.transpose(x).shape
    (5, 4, 3)
    >>> np.swapaxes(x, 0, -1).shape
    (5, 4, 3)
    >>> np.moveaxis(x, [0, 1], [-1, -2]).shape
    (5, 4, 3)
    >>> np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
    (5, 4, 3)
    """
    pass