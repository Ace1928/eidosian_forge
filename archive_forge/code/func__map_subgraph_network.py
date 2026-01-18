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
def _map_subgraph_network(inputs, outputs):
    """Returns the nodes and layers in the topology from `inputs` to `outputs`.

  Args:
    inputs: List of input tensors.
    outputs: List of output tensors.

  Returns:
    A tuple of List{Node] and List[Layer].
  """
    if not ops.executing_eagerly_outside_functions():
        base_layer_utils.create_keras_history(outputs)
    _, nodes_by_depth, layers, _ = _map_graph_network(inputs, outputs)
    return (nest.flatten([nodes for nodes in nodes_by_depth.values()]), layers)