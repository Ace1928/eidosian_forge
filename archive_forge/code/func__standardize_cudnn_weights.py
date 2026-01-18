import tensorflow as tf
import tree
from keras.src.utils.nest import pack_sequence_as
def _standardize_cudnn_weights(weights, biases, shape, transpose_weights=False):
    """Utility function convert variable to cuDNN compatible parameter.

    Note that Keras weights for kernels are different from the cuDNN format.
    Eg.:

    ```
      Keras                 cuDNN
      [[0, 1, 2],  <--->  [[0, 2, 4],
       [3, 4, 5]]          [1, 3, 5]]
    ```

    If the input weights need to be in a unified format, then set
    `transpose_weights=True` to convert the weights.

    Args:
        weights: list of weights for the kernels and recurrent kernels.
        biases: list of biases for individual gate.
        shape: the shape for the converted variables that will be feed to cuDNN.
        transpose_weights: boolean, whether to transpose the weights.

    Returns:
        The converted weights that can be feed to cuDNN ops as param.
    """

    def convert(w):
        return tf.transpose(w) if transpose_weights else w
    weights = [tf.reshape(convert(x), shape) for x in weights]
    biases = [tf.reshape(x, shape) for x in biases]
    return tf.concat(weights + biases, axis=0)