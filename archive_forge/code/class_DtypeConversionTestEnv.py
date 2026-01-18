import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.weak_tensor import WeakTensor
class DtypeConversionTestEnv:
    """Test environment for different dtype conversion semantics."""

    def __init__(self, promo_mode):
        self._old_promo_mode = ops.promo_mode_enum_to_string(ops.get_dtype_conversion_mode())
        self._new_promo_mode = promo_mode

    def __enter__(self):
        ops.set_dtype_conversion_mode(self._new_promo_mode)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        ops.set_dtype_conversion_mode(self._old_promo_mode)