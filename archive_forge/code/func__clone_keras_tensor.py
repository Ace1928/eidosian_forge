import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import input_layer as input_layer_module
from keras.src.engine import keras_tensor
from keras.src.engine import node as node_module
def _clone_keras_tensor(kt):
    """Create an identical keras_tensor based on the input.

    We use keras_tensor_to_placeholder and keras_tensor_from_tensor to make sure
    inferred shape are not lost during the copy.

    Args:
      kt: the input KerasTensor.

    Returns:
      An identical copy of the input KerasTensor.
    """
    with backend._scratch_graph() as scratch_graph:
        with scratch_graph.as_default():
            placeholder = keras_tensor.keras_tensor_to_placeholder(kt)
            return keras_tensor.keras_tensor_from_tensor(placeholder)