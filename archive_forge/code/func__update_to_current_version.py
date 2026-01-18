import re
import types
import warnings
import tensorflow.compat.v1.logging as logging
import tensorflow.compat.v2 as tf
from google.protobuf import message
from keras.src import backend
from keras.src import regularizers
from keras.src.engine import input_spec
from keras.src.optimizers.legacy import optimizer_v2
from keras.protobuf import saved_metadata_pb2
from keras.protobuf import versions_pb2
from keras.src.saving import object_registration
from keras.src.saving.legacy import model_config
from keras.src.saving.legacy import saving_utils
from keras.src.saving.legacy import serialization
from keras.src.saving.legacy.saved_model import constants
from keras.src.saving.legacy.saved_model import json_utils
from keras.src.saving.legacy.saved_model import utils
from keras.src.saving.legacy.saved_model.serialized_attributes import (
from keras.src.utils import layer_utils
from keras.src.utils import metrics_utils
from keras.src.utils import tf_inspect
from keras.src.utils.generic_utils import LazyLoader
def _update_to_current_version(metadata):
    """Applies version updates to the metadata proto for backwards compat."""
    for node in metadata.nodes:
        if node.version.producer == 1 and node.identifier in [constants.MODEL_IDENTIFIER, constants.SEQUENTIAL_IDENTIFIER, constants.NETWORK_IDENTIFIER]:
            node_metadata = json_utils.decode(node.metadata)
            save_spec = node_metadata.get('save_spec')
            if save_spec is not None:
                node_metadata['full_save_spec'] = ([save_spec], {})
                node.metadata = json_utils.Encoder().encode(node_metadata)
    return metadata