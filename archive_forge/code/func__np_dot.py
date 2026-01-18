def _np_dot(a, b, out=None):
    """
    Dot product of two arrays. Specifically,

    - If both `a` and `b` are 1-D arrays, it is inner product of vectors

    - If both `a` and `b` are 2-D arrays, it is matrix multiplication,

    - If either `a` or `b` is 0-D (scalar), it is equivalent to :func:`multiply`
      and using ``np.multiply(a, b)`` or ``a * b`` is preferred.

    - If `a` is an N-D array and `b` is a 1-D array, it is a sum product over
      the last axis of `a` and `b`.

    - If `a` is an N-D array and `b` is a 2-D array, it is a
      sum product over the last axis of `a` and the second-to-last axis of `b`::

        dot(a, b)[i,j,k] = sum(a[i,j,:] * b[:,k])

    Parameters
    ----------
    a : ndarray
        First argument.
    b : ndarray
        Second argument.

    out : ndarray, optional
        Output argument. It must have the same shape and type as the expected output.

    Returns
    -------
    output : ndarray
        Returns the dot product of `a` and `b`.  If `a` and `b` are both
        scalars or both 1-D arrays then a scalar is returned; otherwise
        an array is returned.
        If `out` is given, then it is returned

    Examples
    --------
    >>> a = np.array(3)
    >>> b = np.array(4)
    >>> np.dot(a, b)
    array(12.)

    For 2-D arrays it is the matrix product:

    >>> a = np.array([[1, 0], [0, 1]])
    >>> b = np.array([[4, 1], [2, 2]])
    >>> np.dot(a, b)
    array([[4., 1.],
           [2., 2.]])

    >>> a = np.arange(3*4*5*6).reshape((3,4,5,6))
    >>> b = np.arange(5*6)[::-1].reshape((6,5))
    >>> np.dot(a, b)[2,3,2,2]
    array(29884.)
    >>> np.sum(a[2,3,2,:] * b[:,2])
    array(29884.)
    """
    pass