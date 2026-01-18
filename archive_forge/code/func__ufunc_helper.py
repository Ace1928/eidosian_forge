import numpy as _np
from ...base import numeric_types, integer_types
from ...util import _sanity_check_params, set_module
from ...util import wrap_np_unary_func, wrap_np_binary_func
from ...context import current_context
from . import _internal as _npi
from . import _api_internal
from ..ndarray import NDArray
def _ufunc_helper(lhs, rhs, fn_array, fn_scalar, lfn_scalar, rfn_scalar=None, out=None):
    """ Helper function for element-wise operation.
    The function will perform numpy-like broadcasting if needed and call different functions.

    Parameters
    --------
    lhs : ndarray or numeric value
        Left-hand side operand.

    rhs : ndarray or numeric value
        Right-hand operand,

    fn_array : function
        Function to be called if both lhs and rhs are of ``ndarray`` type.

    fn_scalar : function
        Function to be called if both lhs and rhs are numeric values.

    lfn_scalar : function
        Function to be called if lhs is ``ndarray`` while rhs is numeric value

    rfn_scalar : function
        Function to be called if lhs is numeric value while rhs is ``ndarray``;
        if none is provided, then the function is commutative, so rfn_scalar is equal to lfn_scalar

    Returns
    --------
    mxnet.numpy.ndarray or scalar
        result array or scalar
    """
    from ...numpy import ndarray
    from ..ndarray import from_numpy
    if isinstance(lhs, numeric_types):
        if isinstance(rhs, numeric_types):
            return fn_scalar(lhs, rhs, out=out)
        elif rfn_scalar is None:
            return lfn_scalar(rhs, float(lhs), out=out)
        else:
            return rfn_scalar(rhs, float(lhs), out=out)
    elif isinstance(rhs, numeric_types):
        return lfn_scalar(lhs, float(rhs), out=out)
    elif isinstance(lhs, ndarray) and isinstance(rhs, ndarray):
        return fn_array(lhs, rhs, out=out)
    else:
        raise TypeError('type {} not supported'.format(str(type(rhs))))