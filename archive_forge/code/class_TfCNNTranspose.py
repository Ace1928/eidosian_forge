from typing import Callable, List, Optional, Tuple, Union
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_tf
class TfCNNTranspose(tf.keras.Model):
    """A model containing a CNNTranspose with N Conv2DTranspose layers.

    All layers share the same activation function, bias setup (use bias or not),
    and LayerNormalization setup (use layer normalization or not), except for the last
    one, which is never activated and never layer norm'd.

    Note that there is no reshaping/flattening nor an additional dense layer at the
    beginning or end of the stack. The input as well as output of the network are 3D
    tensors of dimensions [width x height x num output filters].
    """

    def __init__(self, *, input_dims: Union[List[int], Tuple[int]], cnn_transpose_filter_specifiers: List[List[Union[int, List]]], cnn_transpose_use_bias: bool=True, cnn_transpose_activation: Optional[str]='relu', cnn_transpose_use_layernorm: bool=False):
        """Initializes a TfCNNTranspose instance.

        Args:
            input_dims: The 3D input dimensions of the network (incoming image).
            cnn_transpose_filter_specifiers: A list of lists, where each item represents
                one Conv2DTranspose layer. Each such Conv2DTranspose layer is further
                specified by the elements of the inner lists. The inner lists follow
                the format: `[number of filters, kernel, stride]` to
                specify a convolutional-transpose layer stacked in order of the
                outer list.
                `kernel` as well as `stride` might be provided as width x height tuples
                OR as single ints representing both dimension (width and height)
                in case of square shapes.
            cnn_transpose_use_bias: Whether to use bias on all Conv2DTranspose layers.
            cnn_transpose_use_layernorm: Whether to insert a LayerNormalization
                functionality in between each Conv2DTranspose layer's outputs and its
                activation.
                The last Conv2DTranspose layer will not be normed, regardless.
            cnn_transpose_activation: The activation function to use after each layer
                (except for the last Conv2DTranspose layer, which is always
                non-activated).
        """
        super().__init__()
        assert len(input_dims) == 3
        cnn_transpose_activation = get_activation_fn(cnn_transpose_activation, framework='tf2')
        layers = []
        layers.append(tf.keras.layers.Input(shape=input_dims))
        for i, (num_filters, kernel_size, strides) in enumerate(cnn_transpose_filter_specifiers):
            is_final_layer = i == len(cnn_transpose_filter_specifiers) - 1
            layers.append(tf.keras.layers.Conv2DTranspose(filters=num_filters, kernel_size=kernel_size, strides=strides, padding='same', activation=None if cnn_transpose_use_layernorm or is_final_layer else cnn_transpose_activation, use_bias=cnn_transpose_use_bias or is_final_layer))
            if cnn_transpose_use_layernorm and (not is_final_layer):
                layers.append(tf.keras.layers.LayerNormalization(axis=[-3, -2, -1], epsilon=1e-05))
                layers.append(tf.keras.layers.Activation(cnn_transpose_activation))
        self.cnn_transpose = tf.keras.Sequential(layers)
        self.expected_input_dtype = tf.float32

    def call(self, inputs, **kwargs):
        return self.cnn_transpose(tf.cast(inputs, self.expected_input_dtype))