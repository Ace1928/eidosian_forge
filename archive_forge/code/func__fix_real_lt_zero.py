import numpy.core.numeric as nx
import numpy.core.numerictypes as nt
from numpy.core.numeric import asarray, any
from numpy.core.overrides import array_function_dispatch
from numpy.lib.type_check import isreal
def _fix_real_lt_zero(x):
    """Convert `x` to complex if it has real, negative components.

    Otherwise, output is just the array version of the input (via asarray).

    Parameters
    ----------
    x : array_like

    Returns
    -------
    array

    Examples
    --------
    >>> np.lib.scimath._fix_real_lt_zero([1,2])
    array([1, 2])

    >>> np.lib.scimath._fix_real_lt_zero([-1,2])
    array([-1.+0.j,  2.+0.j])

    """
    x = asarray(x)
    if any(isreal(x) & (x < 0)):
        x = _tocomplex(x)
    return x