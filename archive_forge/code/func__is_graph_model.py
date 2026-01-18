import numpy as np
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
def _is_graph_model(layer):
    return hasattr(layer, '_is_graph_network') and layer._is_graph_network or layer.__class__.__name__ == 'Sequential'