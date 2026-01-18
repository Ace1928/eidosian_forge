import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class AltCLIPConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`AltCLIPModel`]. It is used to instantiate an
    AltCLIP model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the AltCLIP
    [BAAI/AltCLIP](https://huggingface.co/BAAI/AltCLIP) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AltCLIPTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`AltCLIPVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 768):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original CLIP implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import AltCLIPConfig, AltCLIPModel

    >>> # Initializing a AltCLIPConfig with BAAI/AltCLIP style configuration
    >>> configuration = AltCLIPConfig()

    >>> # Initializing a AltCLIPModel (with random weights) from the BAAI/AltCLIP style configuration
    >>> model = AltCLIPModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a AltCLIPConfig from a AltCLIPTextConfig and a AltCLIPVisionConfig

    >>> # Initializing a AltCLIPText and AltCLIPVision configuration
    >>> config_text = AltCLIPTextConfig()
    >>> config_vision = AltCLIPVisionConfig()

    >>> config = AltCLIPConfig.from_text_vision_configs(config_text, config_vision)
    ```"""
    model_type = 'altclip'

    def __init__(self, text_config=None, vision_config=None, projection_dim=768, logit_scale_init_value=2.6592, **kwargs):
        text_config_dict = kwargs.pop('text_config_dict', None)
        vision_config_dict = kwargs.pop('vision_config_dict', None)
        super().__init__(**kwargs)
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}
            _text_config_dict = AltCLIPTextConfig(**text_config_dict).to_dict()
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and (key not in ['transformers_version']):
                    if key in text_config_dict:
                        message = f'`{key}` is found in both `text_config_dict` and `text_config` but with different values. The value `text_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`text_config_dict` is provided which will be used to initialize `AltCLIPTextConfig`. The value `text_config["{key}"]` will be overriden.'
                    logger.info(message)
            text_config.update(_text_config_dict)
        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}
            _vision_config_dict = AltCLIPVisionConfig(**vision_config_dict).to_dict()
            if 'id2label' in _vision_config_dict:
                _vision_config_dict['id2label'] = {str(key): value for key, value in _vision_config_dict['id2label'].items()}
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and (key not in ['transformers_version']):
                    if key in vision_config_dict:
                        message = f'`{key}` is found in both `vision_config_dict` and `vision_config` but with different values. The value `vision_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`vision_config_dict` is provided which will be used to initialize `AltCLIPVisionConfig`. The value `vision_config["{key}"]` will be overriden.'
                    logger.info(message)
            vision_config.update(_vision_config_dict)
        if text_config is None:
            text_config = {}
            logger.info('`text_config` is `None`. Initializing the `AltCLIPTextConfig` with default values.')
        if vision_config is None:
            vision_config = {}
            logger.info('`vision_config` is `None`. initializing the `AltCLIPVisionConfig` with default values.')
        self.text_config = AltCLIPTextConfig(**text_config)
        self.vision_config = AltCLIPVisionConfig(**vision_config)
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0

    @classmethod
    def from_text_vision_configs(cls, text_config: AltCLIPTextConfig, vision_config: AltCLIPVisionConfig, **kwargs):
        """
        Instantiate a [`AltCLIPConfig`] (or a derived class) from altclip text model configuration and altclip vision
        model configuration.

        Returns:
            [`AltCLIPConfig`]: An instance of a configuration object
        """
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)