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
def _validate_mode(self, mode):
    """Make sure that mode can be run using the SavedModel."""
    if mode not in self._available_modes:
        raise RuntimeError('%s mode is not available in the SavedModel. Use saved_model_cli to check that the Metagraph for this mode has been exported.' % mode)