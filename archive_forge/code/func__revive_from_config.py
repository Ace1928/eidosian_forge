import os
import re
import types
from google.protobuf import message
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.protobuf import saved_metadata_pb2
from tensorflow.python.keras.protobuf import versions_pb2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.saving.saved_model.serialized_attributes import CommonEndpoints
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load as tf_load
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.saved_model import revived_types
from tensorflow.python.trackable import base as trackable
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import compat
from tensorflow.python.util import nest
def _revive_from_config(self, identifier, metadata, node_id):
    """Revives a layer/model from config, or returns None."""
    if identifier == constants.METRIC_IDENTIFIER:
        obj = self._revive_metric_from_config(metadata)
    else:
        obj = self._revive_graph_network(identifier, metadata, node_id) or self._revive_layer_or_model_from_config(metadata, node_id)
    if obj is None:
        return (None, None)
    setter = self._config_node_setter(_revive_setter)
    self._add_children_recreated_from_config(obj, self._proto.nodes[node_id], node_id)
    return (obj, setter)