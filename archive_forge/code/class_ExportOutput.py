import abc
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.saving.utils_v1 import signature_def_utils as unexported_signature_utils
from tensorflow.python.saved_model import signature_def_utils
class ExportOutput(object):
    """Represents an output of a model that can be served.

  These typically correspond to model heads.
  """
    __metaclass__ = abc.ABCMeta
    _SEPARATOR_CHAR = '/'

    @abc.abstractmethod
    def as_signature_def(self, receiver_tensors):
        """Generate a SignatureDef proto for inclusion in a MetaGraphDef.

    The SignatureDef will specify outputs as described in this ExportOutput,
    and will use the provided receiver_tensors as inputs.

    Args:
      receiver_tensors: a `Tensor`, or a dict of string to `Tensor`, specifying
        input nodes that will be fed.
    """
        pass

    def _check_output_key(self, key, error_label):
        if isinstance(key, tuple):
            key = self._SEPARATOR_CHAR.join(key)
        if not isinstance(key, str):
            raise ValueError('{} output key must be a string; got {}.'.format(error_label, key))
        return key

    def _wrap_and_check_outputs(self, outputs, single_output_default_name, error_label=None):
        """Wraps raw tensors as dicts and checks type.

    Note that we create a new dict here so that we can overwrite the keys
    if necessary.

    Args:
      outputs: A `Tensor` or a dict of string to `Tensor`.
      single_output_default_name: A string key for use in the output dict
        if the provided `outputs` is a raw tensor.
      error_label: descriptive string for use in error messages. If none,
        single_output_default_name will be used.

    Returns:
      A dict of tensors

    Raises:
      ValueError: if the outputs dict keys are not strings or tuples of strings
        or the values are not Tensors.
    """
        if not isinstance(outputs, dict):
            outputs = {single_output_default_name: outputs}
        output_dict = {}
        for key, value in outputs.items():
            error_name = error_label or single_output_default_name
            key = self._check_output_key(key, error_name)
            if not isinstance(value, tensor.Tensor):
                raise ValueError('{} output value must be a Tensor; got {}.'.format(error_name, value))
            output_dict[key] = value
        return output_dict