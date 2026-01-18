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
def _extract_available_modes(self):
    """Return list of modes found in SavedModel."""
    available_modes = []
    tf.compat.v1.logging.info('Checking available modes for SavedModelEstimator.')
    for mode in [ModeKeys.TRAIN, ModeKeys.EVAL, ModeKeys.PREDICT]:
        try:
            self._get_meta_graph_def_for_mode(mode)
        except RuntimeError:
            tf.compat.v1.logging.warn('%s mode not found in SavedModel.' % mode)
            continue
        if self._get_signature_def_for_mode(mode) is not None:
            available_modes.append(mode)
    tf.compat.v1.logging.info('Available modes for Estimator: %s' % available_modes)
    return available_modes