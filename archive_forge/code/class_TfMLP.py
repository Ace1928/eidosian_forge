from typing import Callable, List, Optional, Tuple, Union
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_tf
class TfMLP(tf.keras.Model):
    """A multi-layer perceptron with N dense layers.

    All layers (except for an optional additional extra output layer) share the same
    activation function, bias setup (use bias or not), and LayerNorm setup
    (use layer normalization or not).

    If `output_dim` (int) is not None, an additional, extra output dense layer is added,
    which might have its own activation function (e.g. "linear"). However, the output
    layer does NOT use layer normalization.
    """

    def __init__(self, *, input_dim: int, hidden_layer_dims: List[int], hidden_layer_use_layernorm: bool=False, hidden_layer_use_bias: bool=True, hidden_layer_activation: Optional[Union[str, Callable]]='relu', output_dim: Optional[int]=None, output_use_bias: bool=True, output_activation: Optional[Union[str, Callable]]='linear'):
        """Initialize a TfMLP object.

        Args:
            input_dim: The input dimension of the network. Must not be None.
            hidden_layer_dims: The sizes of the hidden layers. If an empty list, only a
                single layer will be built of size `output_dim`.
            hidden_layer_use_layernorm: Whether to insert a LayerNormalization
                functionality in between each hidden layer's output and its activation.
            hidden_layer_use_bias: Whether to use bias on all dense layers (excluding
                the possible separate output layer).
            hidden_layer_activation: The activation function to use after each layer
                (except for the output). Either a tf.nn.[activation fn] callable or a
                string that's supported by tf.keras.layers.Activation(activation=...),
                e.g. "relu", "ReLU", "silu", or "linear".
            output_dim: The output dimension of the network. If None, no specific output
                layer will be added and the last layer in the stack will have
                size=`hidden_layer_dims[-1]`.
            output_use_bias: Whether to use bias on the separate output layer,
                if any.
            output_activation: The activation function to use for the output layer
                (if any). Either a tf.nn.[activation fn] callable or a string that's
                supported by tf.keras.layers.Activation(activation=...), e.g. "relu",
                "ReLU", "silu", or "linear".
        """
        super().__init__()
        assert input_dim > 0
        layers = []
        layers.append(tf.keras.Input(shape=(input_dim,)))
        hidden_activation = get_activation_fn(hidden_layer_activation, framework='tf2')
        for i in range(len(hidden_layer_dims)):
            layers.append(tf.keras.layers.Dense(hidden_layer_dims[i], activation=hidden_activation if not hidden_layer_use_layernorm else None, use_bias=hidden_layer_use_bias))
            if hidden_layer_use_layernorm:
                layers.append(tf.keras.layers.LayerNormalization(epsilon=1e-05))
                layers.append(tf.keras.layers.Activation(hidden_activation))
        if output_dim is not None:
            output_activation = get_activation_fn(output_activation, framework='tf2')
            layers.append(tf.keras.layers.Dense(output_dim, activation=output_activation, use_bias=output_use_bias))
        self.network = tf.keras.Sequential(layers)

    def call(self, inputs, **kwargs):
        return self.network(inputs)