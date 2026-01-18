import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.engine import base_preprocessing_layer
from keras.src.layers.preprocessing import preprocessing_utils as utils
from keras.src.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export
def _check_input_shape_and_type(self, inputs):
    first_shape = inputs[0].shape.as_list()
    rank = len(first_shape)
    if rank > 2 or (rank == 2 and first_shape[-1] != 1):
        raise ValueError(f'All `HashedCrossing` inputs should have shape `[]`, `[batch_size]` or `[batch_size, 1]`. Received: inputs={inputs}')
    if not all((x.shape.as_list() == first_shape for x in inputs[1:])):
        raise ValueError(f'All `HashedCrossing` inputs should have equal shape. Received: inputs={inputs}')
    if any((isinstance(x, (tf.RaggedTensor, tf.SparseTensor)) for x in inputs)):
        raise ValueError(f'All `HashedCrossing` inputs should be dense tensors. Received: inputs={inputs}')
    if not all((x.dtype.is_integer or x.dtype == tf.string for x in inputs)):
        raise ValueError(f'All `HashedCrossing` inputs should have an integer or string dtype. Received: inputs={inputs}')