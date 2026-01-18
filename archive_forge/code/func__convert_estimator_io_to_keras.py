from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import os
import re
from absl import logging
import tensorflow as tf
from tensorflow.python.checkpoint import checkpoint as trackable_util
from tensorflow_estimator.python.estimator import estimator as estimator_lib
from tensorflow_estimator.python.estimator import model_fn as model_fn_lib
from tensorflow_estimator.python.estimator.export import export_lib
from tensorflow_estimator.python.estimator.mode_keys import ModeKeys
def _convert_estimator_io_to_keras(keras_model, features, labels):
    """Converts estimator features and labels to keras input and target tensors.

  Args:
    keras_model: a compiled `tf.keras.Model` instance, used to determine the
      order of the returned lists.
    features: Dict of tensors or `None`.
    labels: Dict of tensors, a single tensor, or `None`.

  Returns:
    Tuple of (
      list of input tensors or `None`,
      list of target tensors or `None`,
      list of sample weight tensors or `None`)
    The order of tensors is determined by the order set in the keras model.
  """

    def _to_ordered_tensor_list(obj, key_order, obj_name, order_name):
        """Convert obj to an ordered list of tensors.

    Args:
      obj: List, dict, or single tensor. May be `None`.
      key_order: List of strings with the order to return (used if obj is a
        dict).
      obj_name: String name of object (e.g. "features" or "labels")
      order_name: String name of the key order (e.g. "inputs" or "outputs")

    Returns:
      List of tensors, or `None`

    Raises:
      KeyError: If obj has invalid keys.
    """
        if obj is None:
            return None
        elif isinstance(obj, (list, tuple)):
            return [_convert_tensor(x) for x in obj]
        elif isinstance(obj, dict):
            different_keys = set(key_order) - set(obj.keys())
            if different_keys:
                raise FormattedKeyError('The dictionary passed into {obj_name} does not cover requested {order_name} keys defined in the keras model.\n\tExpected keys: {order_keys}\n\t{obj_name} keys: {obj_keys}\n\tMissed keys: {different_keys}'.format(order_name=order_name, order_keys=set(key_order), obj_name=obj_name, obj_keys=set(obj.keys()), different_keys=different_keys))
            return [_convert_tensor(obj[key]) for key in key_order]
        else:
            return [_convert_tensor(obj)]
    features, sample_weight_tensors = _extract_sample_weight_tensors(features)
    input_names = None
    output_names = None
    if isinstance(features, dict):
        input_names = keras_model.input_names if keras_model._is_graph_network else ['input_%d' % i for i in range(1, len(features) + 1)]
    if isinstance(labels, dict):
        output_names = keras_model.output_names if keras_model._is_graph_network else ['output_%d' % i for i in range(1, len(labels) + 1)]
    if isinstance(keras_model.inputs, dict):
        input_tensors = {k: _convert_tensor(features[k]) for k, v in keras_model.inputs.items()}
    elif keras_model.inputs is None and isinstance(features, dict):
        input_tensors = {k: _convert_tensor(v) for k, v in features.items()}
    else:
        input_tensors = _to_ordered_tensor_list(features, input_names, 'features', 'inputs')
    target_tensors = _to_ordered_tensor_list(labels, output_names, 'labels', 'outputs')
    return (input_tensors, target_tensors, sample_weight_tensors)