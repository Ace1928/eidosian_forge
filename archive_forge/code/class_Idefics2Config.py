import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
class Idefics2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`Idefics2Model`]. It is used to instantiate a
    Idefics2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the model of the Idefics2
    [HuggingFaceM4/idefics2-8b](https://huggingface.co/HuggingFaceM4/idefics2-8b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should cache the key/value pairs of the attention mechanism.
        image_token_id (`int`, *optional*, defaults to 32001):
            The id of the "image" token.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether or not to tie the word embeddings with the token embeddings.
        vision_config (`IdeficsVisionConfig` or `dict`, *optional*):
            Custom vision config or dict
        perceiver_config (`IdeficsPerceiverConfig` or `dict`, *optional*):
            Custom perceiver config or dict
        text_config (`MistralConfig` or `dict`, *optional*):
            Custom text config or dict for the text model

    Example:
    ```python
    >>> from transformers import Idefics2Model, Idefics2Config
    >>> # Initializing configuration
    >>> configuration = Idefics2Config()
    >>> # Initializing a model from the configuration
    >>> model = Idefics2Model(configuration)
    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = 'idefics2'
    is_composition = True

    def __init__(self, use_cache=True, image_token_id=32001, tie_word_embeddings=False, vision_config=None, perceiver_config=None, text_config=None, **kwargs):
        self.image_token_id = image_token_id
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        if perceiver_config is None:
            self.perceiver_config = Idefics2PerceiverConfig()
            logger.info('perciver_config is None, using default perceiver config')
        elif isinstance(perceiver_config, dict):
            self.perceiver_config = Idefics2PerceiverConfig(**perceiver_config)
        elif isinstance(perceiver_config, Idefics2PerceiverConfig):
            self.perceiver_config = perceiver_config
        if vision_config is None:
            self.vision_config = Idefics2VisionConfig()
            logger.info('vision_config is None, using default vision config')
        elif isinstance(vision_config, dict):
            self.vision_config = Idefics2VisionConfig(**vision_config)
        elif isinstance(vision_config, Idefics2VisionConfig):
            self.vision_config = vision_config
        if isinstance(text_config, dict):
            text_config['model_type'] = text_config['model_type'] if 'model_type' in text_config else 'mistral'
            text_config = CONFIG_MAPPING[text_config['model_type']](**text_config)
        elif text_config is None:
            logger.info('text_config is None, using default text config')
            text_config = CONFIG_MAPPING['mistral'](max_position_embeddings=4096 * 8, rms_norm_eps=1e-05, pad_token_id=0, tie_word_embeddings=False)
        self.text_config = text_config
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)