import functools
import numpy as _np
from . import numpy as mx_np  # pylint: disable=reimported
from .numpy.multiarray import _NUMPY_ARRAY_FUNCTION_DICT, _NUMPY_ARRAY_UFUNC_DICT
@with_array_function_protocol
def _register_array_function():
    """Register __array_function__ protocol for mxnet.numpy operators so that
    ``mxnet.numpy.ndarray`` can be fed into the official NumPy operators and
    dispatched to MXNet implementation.

    Notes
    -----
    According the __array_function__ protocol (see the following reference),
    there are three kinds of operators that cannot be dispatched using this
    protocol:
    1. Universal functions, which already have their own protocol in the official
    NumPy package.
    2. Array creation functions.
    3. Dispatch for methods of any kind, e.g., methods on np.random.RandomState objects.

    References
    ----------
    https://numpy.org/neps/nep-0018-array-function-protocol.html
    """
    dup = _find_duplicate(_NUMPY_ARRAY_FUNCTION_LIST)
    if dup is not None:
        raise ValueError('Duplicate operator name {} in _NUMPY_ARRAY_FUNCTION_LIST'.format(dup))
    for op_name in _NUMPY_ARRAY_FUNCTION_LIST:
        strs = op_name.split('.')
        if len(strs) == 1:
            mx_np_op = getattr(mx_np, op_name)
            onp_op = getattr(_np, op_name)
            setattr(mx_np, op_name, _implements(onp_op)(mx_np_op))
        elif len(strs) == 2:
            mx_np_submodule = getattr(mx_np, strs[0])
            mx_np_op = getattr(mx_np_submodule, strs[1])
            onp_submodule = getattr(_np, strs[0])
            onp_op = getattr(onp_submodule, strs[1])
            setattr(mx_np_submodule, strs[1], _implements(onp_op)(mx_np_op))
        else:
            raise ValueError('Does not support registering __array_function__ protocol for operator {}'.format(op_name))