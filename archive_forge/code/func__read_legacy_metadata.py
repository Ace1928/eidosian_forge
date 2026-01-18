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
def _read_legacy_metadata(object_graph_def, metadata):
    """Builds a KerasMetadata proto from the SavedModel ObjectGraphDef."""
    node_paths = _generate_object_paths(object_graph_def)
    for node_id, proto in enumerate(object_graph_def.nodes):
        if proto.WhichOneof('kind') == 'user_object' and proto.user_object.identifier in constants.KERAS_OBJECT_IDENTIFIERS:
            if not proto.user_object.metadata:
                raise ValueError('Unable to create a Keras model from this SavedModel. This SavedModel was created with `tf.saved_model.save`, and lacks the Keras metadata.Please save your Keras model by calling `model.save`or `tf.keras.models.save_model`.')
            metadata.nodes.add(node_id=node_id, node_path=node_paths[node_id], version=versions_pb2.VersionDef(producer=1, min_consumer=1, bad_consumers=[]), identifier=proto.user_object.identifier, metadata=proto.user_object.metadata)