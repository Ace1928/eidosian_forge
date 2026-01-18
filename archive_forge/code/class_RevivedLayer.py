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
class RevivedLayer(object):
    """Keras layer loaded from a SavedModel."""

    @classmethod
    def _init_from_metadata(cls, metadata):
        """Create revived layer from metadata stored in the SavedModel proto."""
        init_args = dict(name=metadata['name'], trainable=metadata['trainable'])
        if metadata.get('dtype') is not None:
            init_args['dtype'] = metadata['dtype']
        if metadata.get('batch_input_shape') is not None:
            init_args['batch_input_shape'] = metadata['batch_input_shape']
        revived_obj = cls(**init_args)
        with utils.no_automatic_dependency_tracking_scope(revived_obj):
            revived_obj._expects_training_arg = metadata['expects_training_arg']
            config = metadata.get('config')
            if generic_utils.validate_config(config):
                revived_obj._config = config
            if metadata.get('input_spec') is not None:
                revived_obj.input_spec = recursively_deserialize_keras_object(metadata['input_spec'], module_objects={'InputSpec': input_spec.InputSpec})
            if metadata.get('activity_regularizer') is not None:
                revived_obj.activity_regularizer = regularizers.deserialize(metadata['activity_regularizer'])
            if metadata.get('_is_feature_layer') is not None:
                revived_obj._is_feature_layer = metadata['_is_feature_layer']
            if metadata.get('stateful') is not None:
                revived_obj.stateful = metadata['stateful']
        return (revived_obj, _revive_setter)

    @property
    def keras_api(self):
        return self._serialized_attributes.get(constants.KERAS_ATTR, None)

    def get_config(self):
        if hasattr(self, '_config'):
            return self._config
        else:
            raise NotImplementedError