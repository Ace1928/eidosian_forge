import abc
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.saving.utils_v1 import signature_def_utils as unexported_signature_utils
from tensorflow.python.saved_model import signature_def_utils
class RegressionOutput(ExportOutput):
    """Represents the output of a regression head."""

    def __init__(self, value):
        """Constructor for `RegressionOutput`.

    Args:
      value: a float `Tensor` giving the predicted values.  Required.

    Raises:
      ValueError: if the value is not a `Tensor` with dtype tf.float32.
    """
        if not (isinstance(value, tensor.Tensor) and value.dtype.is_floating):
            raise ValueError('Regression output value must be a float32 Tensor; got {}'.format(value))
        self._value = value

    @property
    def value(self):
        return self._value

    def as_signature_def(self, receiver_tensors):
        if len(receiver_tensors) != 1:
            raise ValueError('Regression input must be a single string Tensor; got {}'.format(receiver_tensors))
        (_, examples), = receiver_tensors.items()
        if dtypes.as_dtype(examples.dtype) != dtypes.string:
            raise ValueError('Regression input must be a single string Tensor; got {}'.format(receiver_tensors))
        return signature_def_utils.regression_signature_def(examples, self.value)