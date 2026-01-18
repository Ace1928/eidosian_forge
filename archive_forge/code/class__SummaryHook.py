from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import shutil
import tempfile
import numpy as np
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.inputs import numpy_io
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
class _SummaryHook(tf.compat.v1.train.SessionRunHook):
    """Saves summaries every N steps."""

    def __init__(self):
        self._summaries = []

    def begin(self):
        self._summary_op = tf.compat.v1.summary.merge_all()

    def before_run(self, run_context):
        return tf.compat.v1.train.SessionRunArgs({'summary': self._summary_op})

    def after_run(self, run_context, run_values):
        s = tf.compat.v1.summary.Summary()
        s.ParseFromString(run_values.results['summary'])
        self._summaries.append(s)

    def summaries(self):
        return tuple(self._summaries)