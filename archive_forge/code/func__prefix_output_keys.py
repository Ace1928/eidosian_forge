import abc
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.saving.utils_v1 import signature_def_utils as unexported_signature_utils
from tensorflow.python.saved_model import signature_def_utils
def _prefix_output_keys(self, output_dict, output_name):
    """Prepend output_name to the output_dict keys if it doesn't exist.

    This produces predictable prefixes for the pre-determined outputs
    of SupervisedOutput.

    Args:
      output_dict: dict of string to Tensor, assumed valid.
      output_name: prefix string to prepend to existing keys.

    Returns:
      dict with updated keys and existing values.
    """
    new_outputs = {}
    for key, val in output_dict.items():
        key = self._prefix_key(key, output_name)
        new_outputs[key] = val
    return new_outputs