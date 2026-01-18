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
def _load_container_state(container, weights_store, assets_store, inner_path, skip_mismatch, visited_trackables):
    used_names = {}
    if isinstance(container, dict):
        container = list(container.values())
    for trackable in container:
        if _is_keras_trackable(trackable):
            if visited_trackables and id(trackable) in visited_trackables:
                continue
            name = generic_utils.to_snake_case(trackable.__class__.__name__)
            if name in used_names:
                used_names[name] += 1
                name = f'{name}_{used_names[name]}'
            else:
                used_names[name] = 0
            _load_state(trackable, weights_store, assets_store, inner_path=tf.io.gfile.join(inner_path, name), skip_mismatch=skip_mismatch, visited_trackables=visited_trackables)