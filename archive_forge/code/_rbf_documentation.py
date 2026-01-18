import numpy as np
from scipy import linalg
from scipy.special import xlogy
from scipy.spatial.distance import cdist, pdist, squareform

    Rbf(*args, **kwargs)

    A class for radial basis function interpolation of functions from
    N-D scattered data to an M-D domain.

    .. legacy:: class

        `Rbf` is legacy code, for new usage please use `RBFInterpolator`
        instead.

    Parameters
    ----------
    *args : arrays
        x, y, z, ..., d, where x, y, z, ... are the coordinates of the nodes
        and d is the array of values at the nodes
    function : str or callable, optional
        The radial basis function, based on the radius, r, given by the norm
        (default is Euclidean distance); the default is 'multiquadric'::

            'multiquadric': sqrt((r/self.epsilon)**2 + 1)
            'inverse': 1.0/sqrt((r/self.epsilon)**2 + 1)
            'gaussian': exp(-(r/self.epsilon)**2)
            'linear': r
            'cubic': r**3
            'quintic': r**5
            'thin_plate': r**2 * log(r)

        If callable, then it must take 2 arguments (self, r). The epsilon
        parameter will be available as self.epsilon. Other keyword
        arguments passed in will be available as well.

    epsilon : float, optional
        Adjustable constant for gaussian or multiquadrics functions
        - defaults to approximate average distance between nodes (which is
        a good start).
    smooth : float, optional
        Values greater than zero increase the smoothness of the
        approximation. 0 is for interpolation (default), the function will
        always go through the nodal points in this case.
    norm : str, callable, optional
        A function that returns the 'distance' between two points, with
        inputs as arrays of positions (x, y, z, ...), and an output as an
        array of distance. E.g., the default: 'euclidean', such that the result
        is a matrix of the distances from each point in ``x1`` to each point in
        ``x2``. For more options, see documentation of
        `scipy.spatial.distances.cdist`.
    mode : str, optional
        Mode of the interpolation, can be '1-D' (default) or 'N-D'. When it is
        '1-D' the data `d` will be considered as 1-D and flattened
        internally. When it is 'N-D' the data `d` is assumed to be an array of
        shape (n_samples, m), where m is the dimension of the target domain.


    Attributes
    ----------
    N : int
        The number of data points (as determined by the input arrays).
    di : ndarray
        The 1-D array of data values at each of the data coordinates `xi`.
    xi : ndarray
        The 2-D array of data coordinates.
    function : str or callable
        The radial basis function. See description under Parameters.
    epsilon : float
        Parameter used by gaussian or multiquadrics functions. See Parameters.
    smooth : float
        Smoothing parameter. See description under Parameters.
    norm : str or callable
        The distance function. See description under Parameters.
    mode : str
        Mode of the interpolation. See description under Parameters.
    nodes : ndarray
        A 1-D array of node values for the interpolation.
    A : internal property, do not use

    See Also
    --------
    RBFInterpolator

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.interpolate import Rbf
    >>> rng = np.random.default_rng()
    >>> x, y, z, d = rng.random((4, 50))
    >>> rbfi = Rbf(x, y, z, d)  # radial basis function interpolator instance
    >>> xi = yi = zi = np.linspace(0, 1, 20)
    >>> di = rbfi(xi, yi, zi)   # interpolated values
    >>> di.shape
    (20,)

    