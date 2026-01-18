import abc
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export
Converts ResourceVariable components to Tensors.

    Override this method to explicitly convert ResourceVariables embedded in the
    CompositeTensor to Tensors. By default, it returns the CompositeTensor
    unchanged.

    Returns:
      A CompositeTensor with all its ResourceVariable components converted to
      Tensors.
    