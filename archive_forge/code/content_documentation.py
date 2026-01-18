from cupy import _core
import cupy
Test element-wise for positive infinity, return result as bool array.

    Parameters
    ----------
    x : cupy.ndarray
        Input array.
    out : cupy.ndarray
        A location into which the result is stored. If provided,
        it should have a shape that input broadcasts to.
        By default, None, a freshly- allocated boolean array,
        is returned.

    Returns
    -------
    y : cupy.ndarray
        Boolean array of same shape as ``x``.

    Examples
    --------
    >>> cupy.isposinf(0)
    array(False)
    >>> cupy.isposinf(cupy.inf)
    array(True)
    >>> cupy.isposinf(cupy.array([-cupy.inf, -4, cupy.nan, 0, 4, cupy.inf]))
    array([False, False, False, False, False,  True])

    See Also
    --------
    numpy.isposinf

    