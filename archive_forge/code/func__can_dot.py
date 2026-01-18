import itertools
import operator
from numpy.core.multiarray import c_einsum
from numpy.core.numeric import asanyarray, tensordot
from numpy.core.overrides import array_function_dispatch
def _can_dot(inputs, result, idx_removed):
    """
    Checks if we can use BLAS (np.tensordot) call and its beneficial to do so.

    Parameters
    ----------
    inputs : list of str
        Specifies the subscripts for summation.
    result : str
        Resulting summation.
    idx_removed : set
        Indices that are removed in the summation


    Returns
    -------
    type : bool
        Returns true if BLAS should and can be used, else False

    Notes
    -----
    If the operations is BLAS level 1 or 2 and is not already aligned
    we default back to einsum as the memory movement to copy is more
    costly than the operation itself.


    Examples
    --------

    # Standard GEMM operation
    >>> _can_dot(['ij', 'jk'], 'ik', set('j'))
    True

    # Can use the standard BLAS, but requires odd data movement
    >>> _can_dot(['ijj', 'jk'], 'ik', set('j'))
    False

    # DDOT where the memory is not aligned
    >>> _can_dot(['ijk', 'ikj'], '', set('ijk'))
    False

    """
    if len(idx_removed) == 0:
        return False
    if len(inputs) != 2:
        return False
    input_left, input_right = inputs
    for c in set(input_left + input_right):
        nl, nr = (input_left.count(c), input_right.count(c))
        if nl > 1 or nr > 1 or nl + nr > 2:
            return False
        if nl + nr - 1 == int(c in result):
            return False
    set_left = set(input_left)
    set_right = set(input_right)
    keep_left = set_left - idx_removed
    keep_right = set_right - idx_removed
    rs = len(idx_removed)
    if input_left == input_right:
        return True
    if set_left == set_right:
        return False
    if input_left[-rs:] == input_right[:rs]:
        return True
    if input_left[:rs] == input_right[-rs:]:
        return True
    if input_left[-rs:] == input_right[-rs:]:
        return True
    if input_left[:rs] == input_right[:rs]:
        return True
    if not keep_left or not keep_right:
        return False
    return True