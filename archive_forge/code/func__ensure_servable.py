import collections
import os
import time
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import op_selector
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.saved_model.model_utils import export_output as export_output_lib
from tensorflow.python.saved_model.model_utils import mode_keys
from tensorflow.python.saved_model.model_utils.mode_keys import KerasModeKeys as ModeKeys
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
def _ensure_servable(input_tensors, names_to_output_tensor_infos):
    """Check that the signature outputs don't depend on unreachable placeholders.

  Args:
    input_tensors: An iterable of `Tensor`s specified as the signature's inputs.
    names_to_output_tensor_infos: An mapping from output names to respective
      `TensorInfo`s corresponding to the signature's output tensors.

  Raises:
    ValueError: If any of the signature's outputs depend on placeholders not
      provided as signature's inputs.
  """
    plain_input_tensors = nest.flatten(input_tensors, expand_composites=True)
    graph = op_selector.get_unique_graph(plain_input_tensors)
    output_tensors = [utils.get_tensor_from_tensor_info(tensor, graph=graph) for tensor in names_to_output_tensor_infos.values()]
    plain_output_tensors = nest.flatten(output_tensors, expand_composites=True)
    dependency_ops = op_selector.get_backward_walk_ops(plain_output_tensors, stop_at_ts=plain_input_tensors)
    fed_tensors = object_identity.ObjectIdentitySet(plain_input_tensors)
    for dependency_op in dependency_ops:
        if _must_be_fed(dependency_op) and (not all((output in fed_tensors for output in dependency_op.outputs))):
            input_tensor_names = [tensor.name for tensor in plain_input_tensors]
            output_tensor_keys = list(names_to_output_tensor_infos.keys())
            output_tensor_names = [tensor.name for tensor in plain_output_tensors]
            dependency_path = op_selector.show_path(dependency_op, plain_output_tensors, plain_input_tensors)
            raise ValueError(f"The signature's input tensors {input_tensor_names} are insufficient to compute its output keys {output_tensor_keys} (respectively, tensors {output_tensor_names}) because of the dependency on `{dependency_op.name}` which is not given as a signature input, as illustrated by the following dependency path: {dependency_path}")