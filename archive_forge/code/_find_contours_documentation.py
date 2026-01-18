import numpy as np
from ._find_contours_cy import _get_contour_segments
from collections import deque
Find iso-valued contours in a 2D array for a given level value.

    Uses the "marching squares" method to compute a the iso-valued contours of
    the input 2D array for a particular level value. Array values are linearly
    interpolated to provide better precision for the output contours.

    Parameters
    ----------
    image : (M, N) ndarray of double
        Input image in which to find contours.
    level : float, optional
        Value along which to find contours in the array. By default, the level
        is set to (max(image) + min(image)) / 2

        .. versionchanged:: 0.18
            This parameter is now optional.
    fully_connected : str, {'low', 'high'}
         Indicates whether array elements below the given level value are to be
         considered fully-connected (and hence elements above the value will
         only be face connected), or vice-versa. (See notes below for details.)
    positive_orientation : str, {'low', 'high'}
         Indicates whether the output contours will produce positively-oriented
         polygons around islands of low- or high-valued elements. If 'low' then
         contours will wind counter- clockwise around elements below the
         iso-value. Alternately, this means that low-valued elements are always
         on the left of the contour. (See below for details.)
    mask : (M, N) ndarray of bool or None
        A boolean mask, True where we want to draw contours.
        Note that NaN values are always excluded from the considered region
        (``mask`` is set to ``False`` wherever ``array`` is ``NaN``).

    Returns
    -------
    contours : list of (K, 2) ndarrays
        Each contour is a ndarray of ``(row, column)`` coordinates along the contour.

    See Also
    --------
    skimage.measure.marching_cubes

    Notes
    -----
    The marching squares algorithm is a special case of the marching cubes
    algorithm [1]_.  A simple explanation is available here:

    http://users.polytech.unice.fr/~lingrand/MarchingCubes/algo.html

    There is a single ambiguous case in the marching squares algorithm: when
    a given ``2 x 2``-element square has two high-valued and two low-valued
    elements, each pair diagonally adjacent. (Where high- and low-valued is
    with respect to the contour value sought.) In this case, either the
    high-valued elements can be 'connected together' via a thin isthmus that
    separates the low-valued elements, or vice-versa. When elements are
    connected together across a diagonal, they are considered 'fully
    connected' (also known as 'face+vertex-connected' or '8-connected'). Only
    high-valued or low-valued elements can be fully-connected, the other set
    will be considered as 'face-connected' or '4-connected'. By default,
    low-valued elements are considered fully-connected; this can be altered
    with the 'fully_connected' parameter.

    Output contours are not guaranteed to be closed: contours which intersect
    the array edge or a masked-off region (either where mask is False or where
    array is NaN) will be left open. All other contours will be closed. (The
    closed-ness of a contours can be tested by checking whether the beginning
    point is the same as the end point.)

    Contours are oriented. By default, array values lower than the contour
    value are to the left of the contour and values greater than the contour
    value are to the right. This means that contours will wind
    counter-clockwise (i.e. in 'positive orientation') around islands of
    low-valued pixels. This behavior can be altered with the
    'positive_orientation' parameter.

    The order of the contours in the output list is determined by the position
    of the smallest ``x,y`` (in lexicographical order) coordinate in the
    contour.  This is a side-effect of how the input array is traversed, but
    can be relied upon.

    .. warning::

       Array coordinates/values are assumed to refer to the *center* of the
       array element. Take a simple example input: ``[0, 1]``. The interpolated
       position of 0.5 in this array is midway between the 0-element (at
       ``x=0``) and the 1-element (at ``x=1``), and thus would fall at
       ``x=0.5``.

    This means that to find reasonable contours, it is best to find contours
    midway between the expected "light" and "dark" values. In particular,
    given a binarized array, *do not* choose to find contours at the low or
    high value of the array. This will often yield degenerate contours,
    especially around structures that are a single array element wide. Instead
    choose a middle value, as above.

    References
    ----------
    .. [1] Lorensen, William and Harvey E. Cline. Marching Cubes: A High
           Resolution 3D Surface Construction Algorithm. Computer Graphics
           (SIGGRAPH 87 Proceedings) 21(4) July 1987, p. 163-170).
           :DOI:`10.1145/37401.37422`

    Examples
    --------
    >>> a = np.zeros((3, 3))
    >>> a[0, 0] = 1
    >>> a
    array([[1., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])
    >>> find_contours(a, 0.5)
    [array([[0. , 0.5],
           [0.5, 0. ]])]
    