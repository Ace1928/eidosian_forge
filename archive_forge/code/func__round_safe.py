import numpy as np
def _round_safe(coords):
    """Round coords while ensuring successive values are less than 1 apart.

    When rounding coordinates for `line_nd`, we want coordinates that are less
    than 1 apart (always the case, by design) to remain less than one apart.
    However, NumPy rounds values to the nearest *even* integer, so:

    >>> np.round([0.5, 1.5, 2.5, 3.5, 4.5])
    array([0., 2., 2., 4., 4.])

    So, for our application, we detect whether the above case occurs, and use
    ``np.floor`` if so. It is sufficient to detect that the first coordinate
    falls on 0.5 and that the second coordinate is 1.0 apart, since we assume
    by construction that the inter-point distance is less than or equal to 1
    and that all successive points are equidistant.

    Parameters
    ----------
    coords : 1D array of float
        The coordinates array. We assume that all successive values are
        equidistant (``np.all(np.diff(coords) = coords[1] - coords[0])``)
        and that this distance is no more than 1
        (``np.abs(coords[1] - coords[0]) <= 1``).

    Returns
    -------
    rounded : 1D array of int
        The array correctly rounded for an indexing operation, such that no
        successive indices will be more than 1 apart.

    Examples
    --------
    >>> coords0 = np.array([0.5, 1.25, 2., 2.75, 3.5])
    >>> _round_safe(coords0)
    array([0, 1, 2, 3, 4])
    >>> coords1 = np.arange(0.5, 8, 1)
    >>> coords1
    array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5])
    >>> _round_safe(coords1)
    array([0, 1, 2, 3, 4, 5, 6, 7])
    """
    if len(coords) > 1 and coords[0] % 1 == 0.5 and (coords[1] - coords[0] == 1):
        _round_function = np.floor
    else:
        _round_function = np.round
    return _round_function(coords).astype(int)