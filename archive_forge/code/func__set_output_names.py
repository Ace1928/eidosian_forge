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
def _set_output_names(self):
    """Assigns unique names to the Network's outputs.

    Output layers with multiple output tensors would otherwise lead to duplicate
    names in self.output_names.
    """
    uniquified = []
    output_names = set()
    prefix_count = {}
    for layer in self._output_layers:
        proposal = layer.name
        while proposal in output_names:
            existing_count = prefix_count.get(layer.name, 1)
            proposal = '{}_{}'.format(layer.name, existing_count)
            prefix_count[layer.name] = existing_count + 1
        output_names.add(proposal)
        uniquified.append(proposal)
    self.output_names = uniquified