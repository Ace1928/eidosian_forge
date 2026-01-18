import numbers
import tensorflow.compat.v2 as tf
from keras.src import backend
from keras.src.engine import base_layer
from keras.src.utils import control_flow_util
from tensorflow.python.util.tf_export import keras_export
Applies Dropout to the input.

    The Dropout layer randomly sets input units to 0 with a frequency of `rate`
    at each step during training time, which helps prevent overfitting.
    Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over
    all inputs is unchanged.

    Note that the Dropout layer only applies when `training` is set to True
    such that no values are dropped during inference. When using `model.fit`,
    `training` will be appropriately set to True automatically, and in other
    contexts, you can set the kwarg explicitly to True when calling the layer.

    (This is in contrast to setting `trainable=False` for a Dropout layer.
    `trainable` does not affect the layer's behavior, as Dropout does
    not have any variables/weights that can be frozen during training.)

    >>> tf.random.set_seed(0)
    >>> layer = tf.keras.layers.Dropout(.2, input_shape=(2,))
    >>> data = np.arange(10).reshape(5, 2).astype(np.float32)
    >>> print(data)
    [[0. 1.]
     [2. 3.]
     [4. 5.]
     [6. 7.]
     [8. 9.]]
    >>> outputs = layer(data, training=True)
    >>> print(outputs)
    tf.Tensor(
    [[ 0.    1.25]
     [ 2.5   3.75]
     [ 5.    6.25]
     [ 7.5   8.75]
     [10.    0.  ]], shape=(5, 2), dtype=float32)

    Args:
      rate: Float between 0 and 1. Fraction of the input units to drop.
      noise_shape: 1D integer tensor representing the shape of the
        binary dropout mask that will be multiplied with the input.
        For instance, if your inputs have shape
        `(batch_size, timesteps, features)` and
        you want the dropout mask to be the same for all timesteps,
        you can use `noise_shape=(batch_size, 1, features)`.
      seed: A Python integer to use as random seed.

    Call arguments:
      inputs: Input tensor (of any rank).
      training: Python boolean indicating whether the layer should behave in
        training mode (adding dropout) or in inference mode (doing nothing).
    