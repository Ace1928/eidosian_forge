import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from ...utils import logging
from ..auto import CONFIG_MAPPING
class Blip2Config(PretrainedConfig):
    """
    [`Blip2Config`] is the configuration class to store the configuration of a [`Blip2ForConditionalGeneration`]. It is
    used to instantiate a BLIP-2 model according to the specified arguments, defining the vision model, Q-Former model
    and language model configs. Instantiating a configuration with the defaults will yield a similar configuration to
    that of the BLIP-2 [Salesforce/blip2-opt-2.7b](https://huggingface.co/Salesforce/blip2-opt-2.7b) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Blip2VisionConfig`].
        qformer_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Blip2QFormerConfig`].
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize any [`PretrainedConfig`].
        num_query_tokens (`int`, *optional*, defaults to 32):
            The number of query tokens passed through the Transformer.

        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import (
    ...     Blip2VisionConfig,
    ...     Blip2QFormerConfig,
    ...     OPTConfig,
    ...     Blip2Config,
    ...     Blip2ForConditionalGeneration,
    ... )

    >>> # Initializing a Blip2Config with Salesforce/blip2-opt-2.7b style configuration
    >>> configuration = Blip2Config()

    >>> # Initializing a Blip2ForConditionalGeneration (with random weights) from the Salesforce/blip2-opt-2.7b style configuration
    >>> model = Blip2ForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Blip2Config from a Blip2VisionConfig, Blip2QFormerConfig and any PretrainedConfig

    >>> # Initializing BLIP-2 vision, BLIP-2 Q-Former and language model configurations
    >>> vision_config = Blip2VisionConfig()
    >>> qformer_config = Blip2QFormerConfig()
    >>> text_config = OPTConfig()

    >>> config = Blip2Config.from_text_vision_configs(vision_config, qformer_config, text_config)
    ```"""
    model_type = 'blip-2'

    def __init__(self, vision_config=None, qformer_config=None, text_config=None, num_query_tokens=32, **kwargs):
        super().__init__(**kwargs)
        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. initializing the Blip2VisionConfig with default values.')
        if qformer_config is None:
            qformer_config = {}
            logger.info('qformer_config is None. Initializing the Blip2QFormerConfig with default values.')
        if text_config is None:
            text_config = {}
            logger.info('text_config is None. Initializing the text config with default values (`OPTConfig`).')
        self.vision_config = Blip2VisionConfig(**vision_config)
        self.qformer_config = Blip2QFormerConfig(**qformer_config)
        text_model_type = text_config['model_type'] if 'model_type' in text_config else 'opt'
        self.text_config = CONFIG_MAPPING[text_model_type](**text_config)
        self.tie_word_embeddings = self.text_config.tie_word_embeddings
        self.is_encoder_decoder = self.text_config.is_encoder_decoder
        self.num_query_tokens = num_query_tokens
        self.qformer_config.encoder_hidden_size = self.vision_config.hidden_size
        self.use_decoder_only_language_model = self.text_config.model_type in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
        self.initializer_factor = 1.0
        self.initializer_range = 0.02

    @classmethod
    def from_vision_qformer_text_configs(cls, vision_config: Blip2VisionConfig, qformer_config: Blip2QFormerConfig, text_config: PretrainedConfig, **kwargs):
        """
        Instantiate a [`Blip2Config`] (or a derived class) from a BLIP-2 vision model, Q-Former and language model
        configurations.

        Returns:
            [`Blip2Config`]: An instance of a configuration object
        """
        return cls(vision_config=vision_config.to_dict(), qformer_config=qformer_config.to_dict(), text_config=text_config.to_dict(), **kwargs)