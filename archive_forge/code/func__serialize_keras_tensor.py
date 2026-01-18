import collections
import copy
import json
import numpy as np
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
def _serialize_keras_tensor(t):
    """Serializes a single Tensor passed to `call`."""
    if hasattr(t, '_keras_history'):
        kh = t._keras_history
        node_index = kh.node_index
        node_key = make_node_key(kh.layer.name, node_index)
        new_node_index = node_conversion_map.get(node_key, 0)
        return [kh.layer.name, new_node_index, kh.tensor_index]
    if isinstance(t, np.ndarray):
        return t.tolist()
    if isinstance(t, tensor_lib.Tensor):
        return backend.get_value(t).tolist()
    return t