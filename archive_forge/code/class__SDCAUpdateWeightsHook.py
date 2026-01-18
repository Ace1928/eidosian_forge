from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.feature_column import feature_column_lib
from tensorflow.python.feature_column import feature_column_v2 as fc_v2
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.canned.linear_optimizer.python.utils import sdca_ops
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import head_utils
from tensorflow_estimator.python.estimator.head import regression_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
class _SDCAUpdateWeightsHook(tf.compat.v1.train.SessionRunHook):
    """SessionRunHook to update and shrink SDCA model weights."""

    def __init__(self, sdca_model, train_op):
        self._sdca_model = sdca_model
        self._train_op = train_op

    def begin(self):
        """Construct the update_weights op.

    The op is implicitly added to the default graph.
    """
        self._update_op = self._sdca_model.update_weights(self._train_op)

    def before_run(self, run_context):
        """Return the update_weights op so that it is executed during this run."""
        return tf.compat.v1.train.SessionRunArgs(self._update_op)