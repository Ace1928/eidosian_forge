from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow_estimator.python.estimator.head import binary_class_head
from tensorflow_estimator.python.estimator.head import multi_class_head
def _assert_simple_summaries(test_case, expected_summaries, summary_str, tol=1e-06):
    """Assert summary the specified simple values.

  Args:
    test_case: test case.
    expected_summaries: Dict of expected tags and simple values.
    summary_str: Serialized `summary_pb2.Summary`.
    tol: Tolerance for relative and absolute.
  """
    summary = tf.compat.v1.summary.Summary()
    summary.ParseFromString(summary_str)
    test_case.assertAllClose(expected_summaries, {v.tag: v.simple_value for v in summary.value}, rtol=tol, atol=tol)