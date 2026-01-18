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
class ReduceSumRegularizer(regularizers.Regularizer):

    def __call__(self, x):
        return math_ops.reduce_sum(x)

    def get_config(self):
        return {}