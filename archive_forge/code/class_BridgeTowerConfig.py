import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class BridgeTowerConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`BridgeTowerModel`]. It is used to instantiate a
    BridgeTower model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the bridgetower-base
    [BridgeTower/bridgetower-base](https://huggingface.co/BridgeTower/bridgetower-base/) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        share_cross_modal_transformer_layers (`bool`, *optional*, defaults to `True`):
            Whether cross modal transformer layers are shared.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler.
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        layer_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the layer normalization layers.
        share_link_tower_layers (`bool`, *optional*, defaults to `False`):
            Whether the bride/link tower layers are shared.
        link_tower_type (`str`, *optional*, defaults to `"add"`):
            Type of the bridge/link layer.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of hidden layers in the Transformer encoder.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether to tie input and output embeddings.
        init_layernorm_from_vision_encoder (`bool`, *optional*, defaults to `False`):
            Whether to init LayerNorm from the vision encoder.
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BridgeTowerTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BridgeTowerVisionConfig`].

    Example:

    ```python
    >>> from transformers import BridgeTowerModel, BridgeTowerConfig

    >>> # Initializing a BridgeTower BridgeTower/bridgetower-base style configuration
    >>> configuration = BridgeTowerConfig()

    >>> # Initializing a model from the BridgeTower/bridgetower-base style configuration
    >>> model = BridgeTowerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'bridgetower'

    def __init__(self, share_cross_modal_transformer_layers=True, hidden_act='gelu', hidden_size=768, initializer_factor=1, layer_norm_eps=1e-05, share_link_tower_layers=False, link_tower_type='add', num_attention_heads=12, num_hidden_layers=6, tie_word_embeddings=False, init_layernorm_from_vision_encoder=False, text_config=None, vision_config=None, **kwargs):
        _ = kwargs.pop('text_config_dict', None)
        _ = kwargs.pop('vision_config_dict', None)
        super().__init__(**kwargs)
        self.share_cross_modal_transformer_layers = share_cross_modal_transformer_layers
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.initializer_factor = initializer_factor
        self.layer_norm_eps = layer_norm_eps
        self.share_link_tower_layers = share_link_tower_layers
        self.link_tower_type = link_tower_type
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.tie_word_embeddings = tie_word_embeddings
        self.init_layernorm_from_vision_encoder = init_layernorm_from_vision_encoder
        if text_config is None:
            text_config = {}
            logger.info('`text_config` is `None`. Initializing the `BridgeTowerTextConfig` with default values.')
        if vision_config is None:
            vision_config = {}
            logger.info('`vision_config` is `None`. Initializing the `BridgeTowerVisionConfig` with default values.')
        self.text_config = BridgeTowerTextConfig(**text_config)
        self.vision_config = BridgeTowerVisionConfig(**vision_config)

    @classmethod
    def from_text_vision_configs(cls, text_config: BridgeTowerTextConfig, vision_config: BridgeTowerVisionConfig, **kwargs):
        """
        Instantiate a [`BridgeTowerConfig`] (or a derived class) from BridgeTower text model configuration. Returns:
            [`BridgeTowerConfig`]: An instance of a configuration object
        """
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)