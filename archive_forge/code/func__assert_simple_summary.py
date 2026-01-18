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
def _assert_simple_summary(testcase, expected_values, actual_summary):
    """Assert summary the specified simple values.

  Args:
    testcase: A TestCase instance.
    expected_values: Dict of expected tags and simple values.
    actual_summary: `summary_pb2.Summary`.
  """
    testcase.assertAllClose(expected_values, {v.tag: v.simple_value for v in actual_summary.value if v.tag in expected_values})