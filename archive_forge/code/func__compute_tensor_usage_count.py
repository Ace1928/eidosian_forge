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
def _compute_tensor_usage_count(self):
    """Compute the #. of tensor usages for all the output tensors of layers.

    The computed tensor usage count is saved as `self._tensor_usage_count`. This
    is later used for saving memory in eager computation by releasing
    no-longer-needed tensors as early as possible.
    """
    tensor_usage_count = collections.Counter()
    available_tensors = set((str(id(tensor)) for tensor in self.inputs))
    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    depth_keys = depth_keys[1:]
    for depth in depth_keys:
        for node in self._nodes_by_depth[depth]:
            input_tensors = {str(id(tensor)) for tensor in nest.flatten(node.keras_inputs)}
            if input_tensors.issubset(available_tensors):
                for tensor in nest.flatten(node.keras_inputs):
                    tensor_usage_count[str(id(tensor))] += 1
                for output_tensor in nest.flatten(node.outputs):
                    available_tensors.add(str(id(output_tensor)))
    for tensor in self.outputs:
        tensor_usage_count[str(id(tensor))] += 1
    self._tensor_usage_count = tensor_usage_count