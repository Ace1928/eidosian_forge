import numpy.core.numeric as nx
import numpy.core.numerictypes as nt
from numpy.core.numeric import asarray, any
from numpy.core.overrides import array_function_dispatch
from numpy.lib.type_check import isreal
def _tocomplex(arr):
    """Convert its input `arr` to a complex array.

    The input is returned as a complex array of the smallest type that will fit
    the original data: types like single, byte, short, etc. become csingle,
    while others become cdouble.

    A copy of the input is always made.

    Parameters
    ----------
    arr : array

    Returns
    -------
    array
        An array with the same input data as the input but in complex form.

    Examples
    --------

    First, consider an input of type short:

    >>> a = np.array([1,2,3],np.short)

    >>> ac = np.lib.scimath._tocomplex(a); ac
    array([1.+0.j, 2.+0.j, 3.+0.j], dtype=complex64)

    >>> ac.dtype
    dtype('complex64')

    If the input is of type double, the output is correspondingly of the
    complex double type as well:

    >>> b = np.array([1,2,3],np.double)

    >>> bc = np.lib.scimath._tocomplex(b); bc
    array([1.+0.j, 2.+0.j, 3.+0.j])

    >>> bc.dtype
    dtype('complex128')

    Note that even if the input was complex to begin with, a copy is still
    made, since the astype() method always copies:

    >>> c = np.array([1,2,3],np.csingle)

    >>> cc = np.lib.scimath._tocomplex(c); cc
    array([1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)

    >>> c *= 2; c
    array([2.+0.j,  4.+0.j,  6.+0.j], dtype=complex64)

    >>> cc
    array([1.+0.j,  2.+0.j,  3.+0.j], dtype=complex64)
    """
    if issubclass(arr.dtype.type, (nt.single, nt.byte, nt.short, nt.ubyte, nt.ushort, nt.csingle)):
        return arr.astype(nt.csingle)
    else:
        return arr.astype(nt.cdouble)