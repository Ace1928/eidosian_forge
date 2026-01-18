
commented out as fdtridfn seems to have bugs and is not in functions.json
see: https://github.com/scipy/scipy/pull/15622#discussion_r811440983

add_newdoc(
    "fdtridfn",
    """
    fdtridfn(p, dfd, x, out=None)

    Inverse to `fdtr` vs dfn

    finds the F density argument dfn such that ``fdtr(dfn, dfd, x) == p``.


    Parameters
    ----------
    p : array_like
        Cumulative probability, in [0, 1].
    dfd : array_like
        Second parameter (positive float).
    x : array_like
        Argument (nonnegative float).
    out : ndarray, optional
        Optional output array for the function values

    Returns
    -------
    dfn : scalar or ndarray
        `dfn` such that ``fdtr(dfn, dfd, x) == p``.

    See Also
    --------
    fdtr, fdtrc, fdtri, fdtridfd


    """)
