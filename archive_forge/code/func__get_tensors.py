from tensorflow.lite.python import util
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader
def _get_tensors(graph, signature_def_tensor_names=None, user_tensor_names=None):
    """Gets the tensors associated with the tensor names.

  Either signature_def_tensor_names or user_tensor_names should be provided. If
  the user provides tensors, the tensors associated with the user provided
  tensor names are provided. Otherwise, the tensors associated with the names in
  the SignatureDef are provided.

  Args:
    graph: GraphDef representing graph.
    signature_def_tensor_names: Tensor names stored in either the inputs or
      outputs of a SignatureDef. (default None)
    user_tensor_names: Tensor names provided by the user. (default None)

  Returns:
    List of tensors.

  Raises:
    ValueError:
      signature_def_tensors and user_tensor_names are undefined or empty.
      user_tensor_names are not valid.
  """
    tensors = []
    if user_tensor_names:
        user_tensor_names = sorted(user_tensor_names)
        tensors = util.get_tensors_from_tensor_names(graph, user_tensor_names)
    elif signature_def_tensor_names:
        tensors = [graph.get_tensor_by_name(name) for name in sorted(signature_def_tensor_names)]
    else:
        raise ValueError('Specify either signature_def_tensor_names or user_tensor_names')
    return tensors