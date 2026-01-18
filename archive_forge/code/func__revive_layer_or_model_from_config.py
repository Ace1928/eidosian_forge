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
def _revive_layer_or_model_from_config(self, metadata, node_id):
    """Revives a layer/custom model from config; returns None if infeasible."""
    class_name = metadata.get('class_name')
    config = metadata.get('config')
    shared_object_id = metadata.get('shared_object_id')
    must_restore_from_config = metadata.get('must_restore_from_config')
    if not generic_utils.validate_config(config):
        return None
    try:
        obj = layers_module.deserialize(generic_utils.serialize_keras_class_and_config(class_name, config, shared_object_id=shared_object_id))
    except ValueError:
        if must_restore_from_config:
            raise RuntimeError('Unable to restore a layer of class {cls}. Layers of class {cls} require that the class be provided to the model loading code, either by registering the class using @keras.utils.register_keras_serializable on the class def and including that file in your program, or by passing the class in a keras.utils.CustomObjectScope that wraps this load call.'.format(cls=class_name))
        else:
            return None
    obj._name = metadata['name']
    if metadata.get('trainable') is not None:
        obj.trainable = metadata['trainable']
    if metadata.get('dtype') is not None:
        obj._set_dtype_policy(metadata['dtype'])
    if metadata.get('stateful') is not None:
        obj.stateful = metadata['stateful']
    if isinstance(obj, training_lib.Model):
        save_spec = metadata.get('save_spec')
        if save_spec is not None:
            obj._set_save_spec(save_spec)
    build_input_shape = metadata.get('build_input_shape')
    built = self._try_build_layer(obj, node_id, build_input_shape)
    if not built:
        return None
    return obj