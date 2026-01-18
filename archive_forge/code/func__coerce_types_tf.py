from importlib import import_module
import autoray as ar
import numpy as np
import semantic_version
from scipy.linalg import block_diag as _scipy_block_diag
from .utils import get_deep_interface
def _coerce_types_tf(tensors):
    """Coerce the dtypes of a list of tensors so that they
    all share the same dtype, without any reduction in information."""
    tf = _i('tf')
    tensors = [tf.convert_to_tensor(t) for t in tensors]
    dtypes = {i.dtype for i in tensors}
    if len(dtypes) == 1:
        return tensors
    complex_priority = [tf.complex64, tf.complex128]
    float_priority = [tf.float16, tf.float32, tf.float64]
    int_priority = [tf.int8, tf.int16, tf.int32, tf.int64]
    complex_type = [i for i in complex_priority if i in dtypes]
    float_type = [i for i in float_priority if i in dtypes]
    int_type = [i for i in int_priority if i in dtypes]
    cast_type = complex_type or float_type or int_type
    cast_type = list(cast_type)[-1]
    return [tf.cast(t, cast_type) for t in tensors]