from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import lookup_ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned import prediction_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _update_accuracy_baseline(self, eval_metrics):
    """Update accuracy baseline metric based on labels mean metric.

    This is the best the model could do by always predicting one class.

    For example, suppose the labels = [0, 1, 0, 1, 1]. So the
    label_mean.total = 3, label_mean.count = 5, and
    label_mean = label_mean.total / label_mean.count = 3 / 5 = 0.6
    By always predicting one class, there are two cases:
    (1) predicted_labels_0 = [0, 0, 0, 0, 0], accuracy_0 = 2 / 5 = 0.4
    (2) predicted_labels_1 = [1, 1, 1, 1, 1], accuracy_1 = 3 / 5 = 0.6
    So the accuracy_baseline = max(accuracy_0, accuracy_1) = 0.6,
                             = max(label_mean, 1 - label_mean)

    To update the total and count of accuracy_baseline,
    accuracy_baseline = max(label_mean, 1 - label_mean)
                      = max(label_mean.total / label_mean.count,
                            1 - label_mean.total / label_mean.count)
                      = max(label_mean.total / label_mean.count,
                      (label_mean.count - label_mean.total) / label_mean.count)
    So accuracy_baseline.total = max(label_mean.total,
                                    (label_mean.count - label_mean.total))
    accuracy_baseline.count = label_mean.count

    Args:
      eval_metrics: A `dict` of metrics to be updated.
    """
    label_mean_metric = eval_metrics[self._label_mean_key]
    accuracy_baseline_metric = eval_metrics[self._accuracy_baseline_key]
    accuracy_baseline_metric.add_update(tf.no_op())
    accuracy_baseline_metric.total = tf.math.maximum(label_mean_metric.total, label_mean_metric.count - label_mean_metric.total)
    accuracy_baseline_metric.count = label_mean_metric.count