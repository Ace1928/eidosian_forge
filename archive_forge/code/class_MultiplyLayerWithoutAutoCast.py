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
class MultiplyLayerWithoutAutoCast(MultiplyLayer):
    """Same as MultiplyLayer, but does not use AutoCastVariables."""

    def build(self, _):
        dtype = self.dtype
        if dtype in ('float16', 'bfloat16'):
            dtype = 'float32'
        self.v = self.add_weight('v', (), initializer='ones', dtype=dtype, experimental_autocast=False, regularizer=self._regularizer)
        self.built = True

    def call(self, inputs):
        self.assert_input_types(inputs)
        assert self.v.dtype in (dtypes.float32, dtypes.float64)
        return self._multiply(inputs, math_ops.cast(self.v, inputs.dtype))