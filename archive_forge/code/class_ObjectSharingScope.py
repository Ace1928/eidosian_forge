import importlib
import inspect
import threading
import types
import warnings
import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.saving import object_registration
from keras.src.saving.legacy import serialization as legacy_serialization
from keras.src.saving.legacy.saved_model.utils import in_tf_saved_model_scope
from keras.src.utils import generic_utils
from tensorflow.python.util import tf_export
from tensorflow.python.util.tf_export import keras_export
class ObjectSharingScope:
    """Scope to enable detection and reuse of previously seen objects."""

    def __enter__(self):
        SHARED_OBJECTS.enabled = True
        SHARED_OBJECTS.id_to_obj_map = {}
        SHARED_OBJECTS.id_to_config_map = {}

    def __exit__(self, *args, **kwargs):
        SHARED_OBJECTS.enabled = False
        SHARED_OBJECTS.id_to_obj_map = {}
        SHARED_OBJECTS.id_to_config_map = {}