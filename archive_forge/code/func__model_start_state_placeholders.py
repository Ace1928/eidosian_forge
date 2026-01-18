from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import tensorflow as tf
from tensorflow_estimator.python.estimator import estimator_lib
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.canned.timeseries import ar_model
from tensorflow_estimator.python.estimator.canned.timeseries import feature_keys
from tensorflow_estimator.python.estimator.canned.timeseries import head as ts_head_lib
from tensorflow_estimator.python.estimator.canned.timeseries import math_utils
from tensorflow_estimator.python.estimator.canned.timeseries import state_management
from tensorflow_estimator.python.estimator.export import export_lib
def _model_start_state_placeholders(self, batch_size_tensor, static_batch_size=None):
    """Creates placeholders with zeroed start state for the current model."""
    gathered_state = {}
    with tf.Graph().as_default():
        self._model.initialize_graph()

        def _zeros_like_constant(tensor):
            return tf.get_static_value(tf.compat.v1.zeros_like(tensor))
        start_state = tf.nest.map_structure(_zeros_like_constant, self._model.get_start_state())
    for prefixed_state_name, state in ts_head_lib.state_to_dictionary(start_state).items():
        state_shape_with_batch = tf.TensorShape((static_batch_size,)).concatenate(state.shape)
        default_state_broadcast = tf.tile(state[None, ...], multiples=tf.concat([batch_size_tensor[None], tf.ones(len(state.shape), dtype=tf.dtypes.int32)], axis=0))
        gathered_state[prefixed_state_name] = tf.compat.v1.placeholder_with_default(input=default_state_broadcast, name=prefixed_state_name, shape=state_shape_with_batch)
    return gathered_state