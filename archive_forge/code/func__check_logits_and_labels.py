from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.export import export_output
from tensorflow_estimator.python.estimator.head import base_head
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _check_logits_and_labels(self, logits, labels=None):
    """Validates the keys of logits and labels."""
    head_names = []
    for head in self._heads:
        head_names.append(head.name)
    if isinstance(logits, dict):
        logits_missing_names = list(set(head_names) - set(list(logits)))
        if logits_missing_names:
            raise ValueError('logits has missing values for head(s): {}'.format(logits_missing_names))
        logits_dict = logits
    else:
        logits_dict = self._split_logits(logits)
    if labels is not None:
        if not isinstance(labels, dict):
            raise ValueError('labels must be a dict. Given: {}'.format(labels))
        labels_missing_names = list(set(head_names) - set(list(labels)))
        if labels_missing_names:
            raise ValueError('labels has missing values for head(s): {}'.format(labels_missing_names))
    return logits_dict