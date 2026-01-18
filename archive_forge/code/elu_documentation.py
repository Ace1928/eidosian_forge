from keras.src import backend
from keras.src.engine.base_layer import Layer
from keras.src.utils import tf_utils
from tensorflow.python.util.tf_export import keras_export
Exponential Linear Unit.

    It follows:

    ```
        f(x) =  alpha * (exp(x) - 1.) for x < 0
        f(x) = x for x >= 0
    ```

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as the input.

    Args:
        alpha: Scale for the negative factor.
    