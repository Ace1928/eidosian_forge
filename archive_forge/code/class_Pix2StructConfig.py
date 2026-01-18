import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class Pix2StructConfig(PretrainedConfig):
    """
    [`Pix2StructConfig`] is the configuration class to store the configuration of a
    [`Pix2StructForConditionalGeneration`]. It is used to instantiate a Pix2Struct model according to the specified
    arguments, defining the text model and vision model configs. Instantiating a configuration with the defaults will
    yield a similar configuration to that of the Pix2Struct-base
    [google/pix2struct-base](https://huggingface.co/google/pix2struct-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Pix2StructTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Pix2StructVisionConfig`].
        initializer_factor (`float`, *optional*, defaults to 1.0):
            Factor to multiply the initialization range with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        is_vqa (`bool`, *optional*, defaults to `False`):
            Whether the model has been fine-tuned for VQA or not.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration

    >>> # Initializing a Pix2StructConfig with google/pix2struct-base style configuration
    >>> configuration = Pix2StructConfig()

    >>> # Initializing a Pix2StructForConditionalGeneration (with random weights) from the google/pix2struct-base style configuration
    >>> model = Pix2StructForConditionalGeneration(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a Pix2StructConfig from a Pix2StructTextConfig and a Pix2StructVisionConfig

    >>> # Initializing a Pix2Struct text and Pix2Struct vision configuration
    >>> config_text = Pix2StructTextConfig()
    >>> config_vision = Pix2StructVisionConfig()

    >>> config = Pix2StructConfig.from_text_vision_configs(config_text, config_vision)
    ```"""
    model_type = 'pix2struct'

    def __init__(self, text_config=None, vision_config=None, initializer_factor=1.0, initializer_range=0.02, is_vqa=False, tie_word_embeddings=False, is_encoder_decoder=True, **kwargs):
        super().__init__(tie_word_embeddings=tie_word_embeddings, is_encoder_decoder=is_encoder_decoder, **kwargs)
        if text_config is None:
            text_config = {}
            logger.info('text_config is None. Initializing the Pix2StructTextConfig with default values.')
        if vision_config is None:
            vision_config = {}
            logger.info('vision_config is None. Initializing the Pix2StructVisionConfig with default values.')
        self.text_config = Pix2StructTextConfig(**text_config)
        self.vision_config = Pix2StructVisionConfig(**vision_config)
        self.decoder_start_token_id = self.text_config.decoder_start_token_id
        self.pad_token_id = self.text_config.pad_token_id
        self.eos_token_id = self.text_config.eos_token_id
        self.initializer_factor = initializer_factor
        self.initializer_range = initializer_range
        self.text_config.initializer_range = self.initializer_range
        self.vision_config.initializer_range = self.initializer_range
        self.is_vqa = is_vqa

    @classmethod
    def from_text_vision_configs(cls, text_config: Pix2StructTextConfig, vision_config: Pix2StructVisionConfig, **kwargs):
        """
        Instantiate a [`Pix2StructConfig`] (or a derived class) from pix2struct text model configuration and pix2struct
        vision model configuration.

        Returns:
            [`Pix2StructConfig`]: An instance of a configuration object
        """
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)