import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class BlipConfig(PretrainedConfig):
    """
    [`BlipConfig`] is the configuration class to store the configuration of a [`BlipModel`]. It is used to instantiate
    a BLIP model according to the specified arguments, defining the text model and vision model configs. Instantiating
    a configuration with the defaults will yield a similar configuration to that of the BLIP-base
    [Salesforce/blip-vqa-base](https://huggingface.co/Salesforce/blip-vqa-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BlipTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`BlipVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 512):
            Dimentionality of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* paramter. Default is used as per the original BLIP implementation.
        image_text_hidden_size (`int`, *optional*, defaults to 256):
            Dimentionality of the hidden state of the image-text fusion layer.
        label_smoothing (float, optional, *optional*, defaults to 0.0):
            A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss, where 0.0 means no smoothing. The targets
            become a mixture of the original ground truth and a uniform distribution as described in
            `Rethinking the Inception Architecture for Computer Vision <https://arxiv.org/abs/1512.00567>`__. Default: :math:`0.0`.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import BlipConfig, BlipModel

    >>> # Initializing a BlipConfig with Salesforce/blip-vqa-base style configuration
    >>> configuration = BlipConfig()

    >>> # Initializing a BlipPModel (with random weights) from the Salesforce/blip-vqa-base style configuration
    >>> model = BlipModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config

    >>> # We can also initialize a BlipConfig from a BlipTextConfig and a BlipVisionConfig

    >>> # Initializing a BLIPText and BLIPVision configuration
    >>> config_text = BlipTextConfig()
    >>> config_vision = BlipVisionConfig()

    >>> config = BlipConfig.from_text_vision_configs(config_text, config_vision)
    ```"""
    model_type = 'blip'

    def __init__(self, text_config=None, vision_config=None, projection_dim=512, logit_scale_init_value=2.6592, image_text_hidden_size=256, label_smoothing=0.0, **kwargs):
        super().__init__(**kwargs)
        if text_config is None:
            text_config = {}
            logger.info('`text_config` is `None`. Initializing the `BlipTextConfig` with default values.')
        if vision_config is None:
            vision_config = {}
            logger.info('`vision_config` is `None`. Initializing the `BlipVisionConfig` with default values.')
        self.text_config = BlipTextConfig(**text_config)
        self.vision_config = BlipVisionConfig(**vision_config)
        self.text_config.encoder_hidden_size = self.vision_config.hidden_size
        self.projection_dim = projection_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_factor = 1.0
        self.initializer_range = 0.02
        self.image_text_hidden_size = image_text_hidden_size
        self.label_smoothing = label_smoothing

    @classmethod
    def from_text_vision_configs(cls, text_config: BlipTextConfig, vision_config: BlipVisionConfig, **kwargs):
        """
        Instantiate a [`BlipConfig`] (or a derived class) from blip text model configuration and blip vision model
        configuration.

        Returns:
            [`BlipConfig`]: An instance of a configuration object
        """
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)