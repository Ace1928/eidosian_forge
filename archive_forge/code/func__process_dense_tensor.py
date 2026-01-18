from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import re
import tensorflow.compat.v2 as tf
from keras.src.engine.base_layer import Layer
from keras.src.saving import serialization_lib
def _process_dense_tensor(self, column, tensor):
    """Reshapes the dense tensor output of a column based on expected shape.

        Args:
          column: A DenseColumn or SequenceDenseColumn object.
          tensor: A dense tensor obtained from the same column.

        Returns:
          Reshaped dense tensor.
        """
    num_elements = column.variable_shape.num_elements()
    target_shape = self._target_shape(tf.shape(tensor), num_elements)
    return tf.reshape(tensor, shape=target_shape)