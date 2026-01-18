import numpy as np
from scipy import signal
Subdivision of polygonal curves using B-Splines.

    Note that the resulting curve is always within the convex hull of the
    original polygon. Circular polygons stay closed after subdivision.

    Parameters
    ----------
    coords : (K, 2) array
        Coordinate array.
    degree : {1, 2, 3, 4, 5, 6, 7}, optional
        Degree of B-Spline. Default is 2.
    preserve_ends : bool, optional
        Preserve first and last coordinate of non-circular polygon. Default is
        False.

    Returns
    -------
    coords : (L, 2) array
        Subdivided coordinate array.

    References
    ----------
    .. [1] http://mrl.nyu.edu/publications/subdiv-course2000/coursenotes00.pdf
    