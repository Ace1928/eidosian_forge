from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
class AssertTypeLayer(base_layer.Layer):
    """A layer which asserts it's inputs are a certain type."""

    def __init__(self, assert_type=None, **kwargs):
        self._assert_type = dtypes.as_dtype(assert_type).name if assert_type else None
        super(AssertTypeLayer, self).__init__(**kwargs)

    def assert_input_types(self, inputs):
        """Asserts `inputs` are of the correct type. Should be called in call()."""
        if self._assert_type:
            inputs_flattened = nest.flatten(inputs)
            for inp in inputs_flattened:
                assert inp.dtype.base_dtype == self._assert_type, 'Input tensor has type %s which does not match assert type %s' % (inp.dtype.name, self._assert_type)