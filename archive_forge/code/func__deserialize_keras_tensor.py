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
def _deserialize_keras_tensor(t):
    """Deserializes a single Keras Tensor passed to `call`."""
    if isinstance(t, tf_utils.ListWrapper):
        t = t.as_list()
        layer_name = t[0]
        node_index = t[1]
        tensor_index = t[2]
        layer = layer_map[layer_name]
        new_node_index = get_node_index(layer, node_index)
        if new_node_index is None:
            raise IndexError
        node = layer._inbound_nodes[new_node_index]
        return nest.flatten(node.outputs)[tensor_index]
    return t