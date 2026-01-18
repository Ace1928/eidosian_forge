
    Create a two-dimensional array with the flattened input as a diagonal.
    Parameters
    ----------
    arr : ndarray
        Input data, which is flattened and set as the `k`-th
        diagonal of the output.
    k : int, optional
        Diagonal to set; 0, the default, corresponds to the "main" diagonal,
        a positive (negative) `k` giving the number of the diagonal above
        (below) the main.
    Returns
    -------
    out : ndarray
        The 2-D output array.
    See Also
    --------
    diag : MATLAB work-alike for 1-D and 2-D arrays.
    diagonal : Return specified diagonals.
    trace : Sum along diagonals.
    Examples
    --------
    >>> np.diagflat([[1,2], [3,4]])
    array([[1, 0, 0, 0],
           [0, 2, 0, 0],
           [0, 0, 3, 0],
           [0, 0, 0, 4]])
    >>> np.diagflat([1,2], 1)
    array([[0, 1, 0],
           [0, 0, 2],
           [0, 0, 0]])
    