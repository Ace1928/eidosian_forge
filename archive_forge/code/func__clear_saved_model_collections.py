from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import six
import tensorflow as tf
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_constants
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _clear_saved_model_collections():
    """Clear collections that are expected empty when exporting a SavedModel.

  The SavedModel builder uses these collections to track ops necessary to
  restore the graph state. These collections are expected to be empty before
  MetaGraphs are added to the builder.
  """
    del tf.compat.v1.get_collection_ref(tf.saved_model.ASSETS_KEY)[:]
    del tf.compat.v1.get_collection_ref(tf.compat.v1.saved_model.LEGACY_INIT_OP_KEY)[:]
    del tf.compat.v1.get_collection_ref(tf.compat.v1.saved_model.MAIN_OP_KEY)[:]
    del tf.compat.v1.get_collection_ref(constants.TRAIN_OP_KEY)[:]