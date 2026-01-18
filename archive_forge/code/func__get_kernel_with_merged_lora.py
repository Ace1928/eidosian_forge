import re
import string
import numpy as np
from keras.src import activations
from keras.src import backend
from keras.src import constraints
from keras.src import dtype_policies
from keras.src import initializers
from keras.src import ops
from keras.src import quantizers
from keras.src import regularizers
from keras.src.api_export import keras_export
from keras.src.layers.input_spec import InputSpec
from keras.src.layers.layer import Layer
def _get_kernel_with_merged_lora(self):
    if isinstance(self.dtype_policy, dtype_policies.QuantizedDTypePolicy):
        kernel_value = self._kernel
        kernel_scale = self.kernel_scale
        if self.lora_enabled:
            kernel_value = ops.divide(kernel_value, kernel_scale)
            kernel_value = ops.add(kernel_value, ops.matmul(self.lora_kernel_a, self.lora_kernel_b))
            kernel_value, kernel_scale = quantizers.abs_max_quantize(kernel_value, axis=self._kernel_reduced_axes)
            kernel_scale = ops.transpose(kernel_scale, self._kernel_transpose_axes)
            if self._kernel_expand_axes:
                kernel_scale = ops.expand_dims(kernel_scale, axis=self._kernel_expand_axes)
            if self._kernel_squeeze_axes:
                kernel_scale = ops.squeeze(kernel_scale, axis=self._kernel_squeeze_axes)
    else:
        kernel_value = self.kernel
        kernel_scale = None
    return (kernel_value, kernel_scale)