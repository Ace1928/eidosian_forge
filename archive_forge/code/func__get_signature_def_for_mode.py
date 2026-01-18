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
def _get_signature_def_for_mode(self, mode):
    meta_graph_def = self._get_meta_graph_def_for_mode(mode)
    if mode == ModeKeys.PREDICT:
        sig_def_key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    else:
        sig_def_key = mode
    if sig_def_key not in meta_graph_def.signature_def:
        tf.compat.v1.logging.warn('Metagraph for mode %s was found, but SignatureDef with key "%s" is missing.' % (mode, sig_def_key))
        return None
    return meta_graph_def.signature_def[sig_def_key]