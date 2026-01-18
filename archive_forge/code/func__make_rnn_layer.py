from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from tensorflow.python.feature_column import feature_column_lib as fc
from tensorflow.python.framework import ops
from tensorflow_estimator.python.estimator import estimator
from tensorflow_estimator.python.estimator import model_fn
from tensorflow_estimator.python.estimator.canned import optimizers
from tensorflow_estimator.python.estimator.estimator_export import estimator_export
from tensorflow_estimator.python.estimator.head import binary_class_head as binary_head_lib
from tensorflow_estimator.python.estimator.head import multi_class_head as multi_head_lib
from tensorflow_estimator.python.estimator.head import sequential_head as seq_head_lib
def _make_rnn_layer(rnn_cell_fn, units, cell_type, return_sequences):
    """Assert arguments are valid and return rnn_layer_fn.

  Args:
    rnn_cell_fn: A function that returns a RNN cell instance that will be used
      to construct the RNN.
    units: Iterable of integer number of hidden units per RNN layer.
    cell_type: A class producing a RNN cell or a string specifying the cell
      type.
    return_sequences: A boolean indicating whether to return the last output
      in the output sequence, or the full sequence.:

  Returns:
    A tf.keras.layers.RNN layer.
  """
    _verify_rnn_cell_input(rnn_cell_fn, units, cell_type)
    if cell_type in _CELL_TYPE_TO_LAYER_MAPPING and isinstance(units, int):
        return _CELL_TYPE_TO_LAYER_MAPPING[cell_type](units=units, return_sequences=return_sequences)
    if not rnn_cell_fn:
        if cell_type == USE_DEFAULT:
            cell_type = _SIMPLE_RNN_KEY
        rnn_cell_fn = _make_rnn_cell_fn(units, cell_type)
    return tf.keras.layers.RNN(cell=rnn_cell_fn(), return_sequences=return_sequences)