def _np_prod(a, axis=None, dtype=None, out=None, keepdims=False):
    """
    Return the product of array elements over a given axis.

    Parameters
    ----------
    a : ndarray
        Input data.
    axis : None or int or tuple of ints, optional
        Axis or axes along which a product is performed.
        The default (`axis` = `None`) is perform a product over all
        the dimensions of the input array. `axis` may be negative, in
        which case it counts from the last to the first axis.
        If this is a tuple of ints, a product is performed on multiple
        axes, instead of a single axis or all the axes as before.
    dtype : data-type, optional
        The data-type of the returned array, as well as of the accumulator
        in which the elements are multiplied.  By default, if `a` is of
        integer type, `dtype` is the default platform integer. (Note: if
        the type of `a` is unsigned, then so is `dtype`.)  Otherwise,
        the dtype is the same as that of `a`.
    out : ndarray, optional
        Alternative output array in which to place the result. It must have
        the same shape as the expected output, but the type of the
        output values will be cast if necessary.
    keepdims : bool, optional
        If this is set to True, the axes which are reduced are left
        in the result as dimensions with size one. With this option,
        the result will broadcast correctly against the original `arr`.

    Returns
    -------
    product_along_axis : ndarray, see `dtype` parameter above.
        An array shaped as `a` but with the specified axis removed.
        Returns a reference to `out` if specified.

    See Also
    --------
    ndarray.prod : equivalent method

    Notes
    -----
    Arithmetic is modular when using integer types, and no error is
    raised on overflow.  That means that, on a 32-bit platform:

    >>> x = np.array([536870910, 536870910, 536870910, 536870910])
    >>> np.prod(x) #random
    array(8.307675e+34)

    Examples
    --------
    By default, calculate the product of all elements:

    >>> np.prod(np.array([1.,2.]))
    array(2.)

    Even when the input array is two-dimensional:

    >>> np.prod(np.array([1.,2.,3.,4.]).reshape((2,2)))
    array(24.)

    But we can also specify the axis over which to multiply:

    >>> np.prod(np.array([1.,2.,3.,4.]).reshape((2,2)), axis=1)
    array([  2.,  12.])

    If the type of `x` is unsigned, then the output type is
    the unsigned platform integer:

    >>> x = np.array([1, 2, 3], dtype=np.uint8)
    >>> np.prod(x).dtype == np.uint8
    True

    If `x` is of a signed integer type, then the output type
    is the default platform integer:

    >>> x = np.array([1, 2, 3], dtype=np.int8)
    >>> np.prod(x).dtype == np.int8
    True
    """
    pass