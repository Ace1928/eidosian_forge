from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
from absl import logging
import tensorflow as tf
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _assert_valid_model(model, custom_objects=None):
    is_subclass = not model._is_graph_network and (not isinstance(model, tf.keras.models.Sequential))
    if is_subclass:
        try:
            custom_objects = custom_objects or {}
            with tf.keras.utils.CustomObjectScope(custom_objects):
                model.__class__.from_config(model.get_config())
        except NotImplementedError:
            raise ValueError('Subclassed `Model`s passed to `model_to_estimator` must implement `Model.get_config` and `Model.from_config`.')