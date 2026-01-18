import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ...utils import logging
from ..auto import CONFIG_MAPPING
class Blip2VisionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Blip2VisionModel`]. It is used to instantiate a
    BLIP-2 vision encoder according to the specified arguments, defining the model architecture. Instantiating a
    configuration defaults will yield a similar configuration to that of the BLIP-2
    [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_size (`int`, *optional*, defaults to 1408):
            Dimensionality of the encoder layers and the pooler layer.
        intermediate_size (`int`, *optional*, defaults to 6144):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 39):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer encoder.
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to 14):
            The size (resolution) of each patch.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` ``"gelu"` are supported. layer_norm_eps (`float`, *optional*, defaults
            to 1e-5): The epsilon used by the layer normalization layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries and values in the self-attention layers.

    Example:

    ```python
    >>> from transformers import Blip2VisionConfig, Blip2VisionModel

    >>> # Initializing a Blip2VisionConfig with Salesforce/blip2-opt-2.7b style configuration
    >>> configuration = Blip2VisionConfig()

    >>> # Initializing a Blip2VisionModel (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
    >>> model = Blip2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'blip_2_vision_model'

    def __init__(self, hidden_size=1408, intermediate_size=6144, num_hidden_layers=39, num_attention_heads=16, image_size=224, patch_size=14, hidden_act='gelu', layer_norm_eps=1e-06, attention_dropout=0.0, initializer_range=1e-10, qkv_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.qkv_bias = qkv_bias

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'blip-2':
            config_dict = config_dict['vision_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)