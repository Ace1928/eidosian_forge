from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import re
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import head as head_lib
from tensorflow_estimator.python.estimator.canned import metric_keys
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.export import export_lib
def _identity_metric_nested(name, input_tensors):
    """Create identity metrics for a nested tuple of Tensors."""
    update_ops = []
    value_tensors = []
    for tensor_number, tensor in enumerate(tf.nest.flatten(input_tensors)):
        value_tensor, update_op = _identity_metric_single(name='{}_{}'.format(name, tensor_number), input_tensor=tensor)
        update_ops.append(update_op)
        value_tensors.append(value_tensor)
    return (tf.nest.pack_sequence_as(input_tensors, value_tensors), tf.group(*update_ops))