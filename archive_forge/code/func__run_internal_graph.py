import collections
import copy
import itertools
import warnings
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_layer as input_layer_module
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import network_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls
def _run_internal_graph(self, inputs, training=None, mask=None):
    """Computes output tensors for new inputs.

    # Note:
        - Can be run on non-Keras tensors.

    Args:
        inputs: Tensor or nested structure of Tensors.
        training: Boolean learning phase.
        mask: (Optional) Tensor or nested structure of Tensors.

    Returns:
        output_tensors
    """
    inputs = self._flatten_to_reference_inputs(inputs)
    if mask is None:
        masks = [None] * len(inputs)
    else:
        masks = self._flatten_to_reference_inputs(mask)
    for input_t, mask in zip(inputs, masks):
        input_t._keras_mask = mask
    tensor_dict = {}
    tensor_usage_count = self._tensor_usage_count
    for x, y in zip(self.inputs, inputs):
        y = self._conform_to_reference_input(y, ref_input=x)
        x_id = str(id(x))
        tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
    nodes_by_depth = self._nodes_by_depth
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    for depth in depth_keys:
        nodes = nodes_by_depth[depth]
        for node in nodes:
            if node.is_input:
                continue
            if any((t_id not in tensor_dict for t_id in node.flat_input_ids)):
                continue
            args, kwargs = node.map_arguments(tensor_dict)
            outputs = node.layer(*args, **kwargs)
            for x_id, y in zip(node.flat_output_ids, nest.flatten(outputs)):
                tensor_dict[x_id] = [y] * tensor_usage_count[x_id]
    output_tensors = []
    for x in self.outputs:
        x_id = str(id(x))
        assert x_id in tensor_dict, 'Could not compute output ' + str(x)
        output_tensors.append(tensor_dict[x_id].pop())
    return nest.pack_sequence_as(self._nested_outputs, output_tensors)