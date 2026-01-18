from collections import OrderedDict
from typing import Mapping
from packaging import version
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

    This is the configuration class to store the configuration of a [`MobileViTV2Model`]. It is used to instantiate a
    MobileViTV2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MobileViTV2
    [apple/mobilevitv2-1.0](https://huggingface.co/apple/mobilevitv2-1.0) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        image_size (`int`, *optional*, defaults to 256):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 2):
            The size (resolution) of each patch.
        expand_ratio (`float`, *optional*, defaults to 2.0):
            Expansion factor for the MobileNetv2 layers.
        hidden_act (`str` or `function`, *optional*, defaults to `"swish"`):
            The non-linear activation function (function or string) in the Transformer encoder and convolution layers.
        conv_kernel_size (`int`, *optional*, defaults to 3):
            The size of the convolutional kernel in the MobileViTV2 layer.
        output_stride (`int`, *optional*, defaults to 32):
            The ratio of the spatial resolution of the output to the resolution of the input image.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for attached classifiers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        aspp_out_channels (`int`, *optional*, defaults to 512):
            Number of output channels used in the ASPP layer for semantic segmentation.
        atrous_rates (`List[int]`, *optional*, defaults to `[6, 12, 18]`):
            Dilation (atrous) factors used in the ASPP layer for semantic segmentation.
        aspp_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the ASPP layer for semantic segmentation.
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.
        n_attn_blocks (`List[int]`, *optional*, defaults to `[2, 4, 3]`):
            The number of attention blocks in each MobileViTV2Layer
        base_attn_unit_dims (`List[int]`, *optional*, defaults to `[128, 192, 256]`):
            The base multiplier for dimensions of attention blocks in each MobileViTV2Layer
        width_multiplier (`float`, *optional*, defaults to 1.0):
            The width multiplier for MobileViTV2.
        ffn_multiplier (`int`, *optional*, defaults to 2):
            The FFN multiplier for MobileViTV2.
        attn_dropout (`float`, *optional*, defaults to 0.0):
            The dropout in the attention layer.
        ffn_dropout (`float`, *optional*, defaults to 0.0):
            The dropout between FFN layers.

    Example:

    ```python
    >>> from transformers import MobileViTV2Config, MobileViTV2Model

    >>> # Initializing a mobilevitv2-small style configuration
    >>> configuration = MobileViTV2Config()

    >>> # Initializing a model from the mobilevitv2-small style configuration
    >>> model = MobileViTV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```