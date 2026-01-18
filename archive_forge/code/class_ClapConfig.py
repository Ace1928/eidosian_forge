import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class ClapConfig(PretrainedConfig):
    """
    [`ClapConfig`] is the configuration class to store the configuration of a [`ClapModel`]. It is used to instantiate
    a CLAP model according to the specified arguments, defining the text model and audio model configs. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the CLAP
    [laion/clap-htsat-fused](https://huggingface.co/laion/clap-htsat-fused) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapTextConfig`].
        audio_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`ClapAudioConfig`].
        logit_scale_init_value (`float`, *optional*, defaults to 14.29):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLAP implementation.
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and audio projection layers.
        projection_hidden_act (`str`, *optional*, defaults to `"relu"`):
            Activation function for the projection layers.
        initializer_factor (`float`, *optional*, defaults to 1.0):
            Factor to scale the initialization of the model weights.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import ClapConfig, ClapModel

    >>> # Initializing a ClapConfig with laion-ai/base style configuration
    >>> configuration = ClapConfig()

    >>> # Initializing a ClapModel (with random weights) from the laion-ai/base style configuration
    >>> model = ClapModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a ClapConfig from a ClapTextConfig and a ClapAudioConfig
    >>> from transformers import ClapTextConfig, ClapAudioConfig

    >>> # Initializing a ClapText and ClapAudioConfig configuration
    >>> config_text = ClapTextConfig()
    >>> config_audio = ClapAudioConfig()

    >>> config = ClapConfig.from_text_audio_configs(config_text, config_audio)
    ```"""
    model_type = 'clap'

    def __init__(self, text_config=None, audio_config=None, logit_scale_init_value=1 / 0.07, projection_dim=512, projection_hidden_act='relu', initializer_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        if text_config is None:
            text_config = {}
            logger.info('text_config is None. Initializing the ClapTextConfig with default values.')
        if audio_config is None:
            audio_config = {}
            logger.info('audio_config is None. initializing the ClapAudioConfig with default values.')
        self.text_config = ClapTextConfig(**text_config)
        self.audio_config = ClapAudioConfig(**audio_config)
        self.text_config.projection_dim = projection_dim
        self.audio_config.projection_dim = projection_dim
        self.text_config.projection_hidden_act = projection_hidden_act
        self.audio_config.projection_hidden_act = projection_hidden_act
        self.projection_dim = projection_dim
        self.projection_hidden_act = projection_hidden_act
        self.hidden_size = self.text_config.hidden_size
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = initializer_factor
        self.num_hidden_layers = self.text_config.num_hidden_layers + len(self.audio_config.depths)

    @classmethod
    def from_text_audio_configs(cls, text_config: ClapTextConfig, audio_config: ClapAudioConfig, **kwargs):
        """
        Instantiate a [`ClapConfig`] (or a derived class) from clap text model configuration and clap audio model
        configuration.

        Returns:
            [`ClapConfig`]: An instance of a configuration object
        """
        return cls(text_config=text_config.to_dict(), audio_config=audio_config.to_dict(), **kwargs)