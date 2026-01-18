import datetime
import io
import json
import os
import re
import tempfile
import threading
import warnings
import zipfile
import numpy as np
import tensorflow.compat.v2 as tf
import keras.src as keras
from keras.src import losses
from keras.src.engine import base_layer
from keras.src.optimizers import optimizer
from keras.src.saving.serialization_lib import ObjectSharingScope
from keras.src.saving.serialization_lib import deserialize_keras_object
from keras.src.saving.serialization_lib import serialize_keras_object
from keras.src.utils import generic_utils
from keras.src.utils import io_utils
def _save_state(trackable, weights_store, assets_store, inner_path, visited_trackables):
    if id(trackable) in visited_trackables:
        return
    if hasattr(trackable, 'save_own_variables') and weights_store:
        trackable.save_own_variables(weights_store.make(inner_path))
    if hasattr(trackable, 'save_assets') and assets_store:
        trackable.save_assets(assets_store.make(inner_path))
    visited_trackables.add(id(trackable))
    for child_attr, child_obj in _walk_trackable(trackable):
        if _is_keras_trackable(child_obj):
            _save_state(child_obj, weights_store, assets_store, inner_path=tf.io.gfile.join(inner_path, child_attr), visited_trackables=visited_trackables)
        elif isinstance(child_obj, (list, dict, tuple, set)):
            _save_container_state(child_obj, weights_store, assets_store, inner_path=tf.io.gfile.join(inner_path, child_attr), visited_trackables=visited_trackables)