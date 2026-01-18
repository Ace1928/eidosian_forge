from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
import os
import tensorflow as tf
from tensorflow_estimator.python.estimator import gc
from tensorflow_estimator.python.estimator import util
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
def _get_best_eval_result(self, event_files):
    """Get the best eval result from event files.

    Args:
      event_files: Absolute pattern of event files.

    Returns:
      The best eval result.
    """
    if not event_files:
        return None
    best_eval_result = None
    for event_file in tf.compat.v1.gfile.Glob(os.path.join(event_files)):
        for event in tf.compat.v1.train.summary_iterator(event_file):
            if event.HasField('summary'):
                event_eval_result = {}
                for value in event.summary.value:
                    if value.HasField('simple_value'):
                        event_eval_result[value.tag] = value.simple_value
                if event_eval_result:
                    if best_eval_result is None or self._compare_fn(best_eval_result, event_eval_result):
                        best_eval_result = event_eval_result
    return best_eval_result