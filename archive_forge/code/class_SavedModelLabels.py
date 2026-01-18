from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
class SavedModelLabels(object):
    """Names of signatures exported with export_saved_model."""
    PREDICT = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    FILTER = 'filter'
    COLD_START_FILTER = 'cold_start_filter'