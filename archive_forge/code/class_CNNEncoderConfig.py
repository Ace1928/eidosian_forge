import abc
from dataclasses import dataclass, field
import functools
from typing import Callable, List, Optional, Tuple, TYPE_CHECKING, Union
import numpy as np
from ray.rllib.models.torch.misc import (
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.annotations import ExperimentalAPI
@ExperimentalAPI
@dataclass
class CNNEncoderConfig(ModelConfig):
    """Configuration for a convolutional (encoder) network.

    The configured CNN encodes 3D-observations into a latent space.
    The stack of layers is composed of a sequence of convolutional layers.
    `input_dims` describes the shape of the input tensor. Beyond that, each layer
    specified by `filter_specifiers` is followed by an activation function according
    to `filter_activation`.

    `output_dims` is reached by either the final convolutional layer's output directly
    OR by flatten this output.

    See ModelConfig for usage details.

    Example:

    .. testcode::

        # Configuration:
        config = CNNEncoderConfig(
            input_dims=[84, 84, 3],  # must be 3D tensor (image: w x h x C)
            cnn_filter_specifiers=[
                [16, [8, 8], 4],
                [32, [4, 4], 2],
            ],
            cnn_activation="relu",
            cnn_use_layernorm=False,
            cnn_use_bias=True,
        )
        model = config.build(framework="torch")

        # Resulting stack in pseudocode:
        # Conv2D(
        #   in_channels=3, out_channels=16,
        #   kernel_size=[8, 8], stride=[4, 4], bias=True,
        # )
        # ReLU()
        # Conv2D(
        #   in_channels=16, out_channels=32,
        #   kernel_size=[4, 4], stride=[2, 2], bias=True,
        # )
        # ReLU()
        # Conv2D(
        #   in_channels=32, out_channels=1,
        #   kernel_size=[1, 1], stride=[1, 1], bias=True,
        # )
        # Flatten()

    Attributes:
        input_dims: The input dimension of the network. These must be given in the
            form of `(width, height, channels)`.
        cnn_filter_specifiers: A list in which each element is another (inner) list
            of either the following forms:
            `[number of channels/filters, kernel, stride]`
            OR:
            `[number of channels/filters, kernel, stride, padding]`, where `padding`
            can either be "same" or "valid".
            When using the first format w/o the `padding` specifier, `padding` is "same"
            by default. Also, `kernel` and `stride` may be provided either as single
            ints (square) or as a tuple/list of two ints (width- and height dimensions)
            for non-squared kernel/stride shapes.
            A good rule of thumb for constructing CNN stacks is:
            When using padding="same", the input "image" will be reduced in size by
            the factor `stride`, e.g. input=(84, 84, 3) stride=2 kernel=x padding="same"
            filters=16 -> output=(42, 42, 16).
            For example, if you would like to reduce an Atari image from its original
            (84, 84, 3) dimensions down to (6, 6, F), you can construct the following
            stack and reduce the w x h dimension of the image by 2 in each layer:
            [[16, 4, 2], [32, 4, 2], [64, 4, 2], [128, 4, 2]] -> output=(6, 6, 128)
        cnn_use_bias: Whether to use bias on all Conv2D layers.
        cnn_activation: The activation function to use after each layer (
            except for the output). The default activation for Conv2d layers is "relu".
        cnn_use_layernorm: Whether to insert a LayerNorm functionality
            in between each CNN layer's output and its activation. Note that
            the output layer.
        flatten_at_end: Whether to flatten the output of the last conv 2D layer into
            a 1D tensor. By default, this is True. Note that if you set this to False,
            you might simply stack another CNNEncoder on top of this one (maybe with
            different activation and bias settings).
    """
    input_dims: Union[List[int], Tuple[int]] = None
    cnn_filter_specifiers: List[List[Union[int, List[int]]]] = field(default_factory=lambda: [[16, [4, 4], 2], [32, [4, 4], 2], [64, [8, 8], 2]])
    cnn_use_bias: bool = True
    cnn_activation: str = 'relu'
    cnn_use_layernorm: bool = False
    flatten_at_end: bool = True

    @property
    def output_dims(self):
        if not self.input_dims:
            return None
        dims = self.input_dims
        for filter_spec in self.cnn_filter_specifiers:
            if len(filter_spec) == 3:
                num_filters, kernel, stride = filter_spec
                padding = 'same'
            else:
                num_filters, kernel, stride, padding = filter_spec
            if padding == 'same':
                _, dims = same_padding(dims[:2], kernel, stride)
            else:
                dims = valid_padding(dims[:2], kernel, stride)
            dims = [dims[0], dims[1], num_filters]
        if self.flatten_at_end:
            return (int(np.prod(dims)),)
        return tuple(dims)

    def _validate(self, framework: str='torch'):
        if len(self.input_dims) != 3:
            raise ValueError(f'`input_dims` ({self.input_dims}) of CNNEncoderConfig must be a 3D tensor (image) with the dimensions meaning: width x height x channels, e.g. `[64, 64, 3]`!')
        if not self.flatten_at_end and len(self.output_dims) != 3:
            raise ValueError(f'`output_dims` ({self.output_dims}) of CNNEncoderConfig must be 3D, e.g. `[4, 4, 128]`, b/c your `flatten_at_end` setting is False! `output_dims` is an inferred value, hence other settings might be wrong.')
        elif self.flatten_at_end and len(self.output_dims) != 1:
            raise ValueError(f'`output_dims` ({self.output_dims}) of CNNEncoderConfig must be 1D, e.g. `[32]`, b/c your `flatten_at_end` setting is True! `output_dims` is an inferred value, hence other settings might be wrong.')

    @_framework_implemented()
    def build(self, framework: str='torch') -> 'Model':
        self._validate(framework)
        if framework == 'torch':
            from ray.rllib.core.models.torch.encoder import TorchCNNEncoder
            return TorchCNNEncoder(self)
        elif framework == 'tf2':
            from ray.rllib.core.models.tf.encoder import TfCNNEncoder
            return TfCNNEncoder(self)