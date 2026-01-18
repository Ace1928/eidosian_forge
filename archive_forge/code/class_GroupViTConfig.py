import os
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging
class GroupViTConfig(PretrainedConfig):
    """
    [`GroupViTConfig`] is the configuration class to store the configuration of a [`GroupViTModel`]. It is used to
    instantiate a GroupViT model according to the specified arguments, defining the text model and vision model
    configs. Instantiating a configuration with the defaults will yield a similar configuration to that of the GroupViT
    [nvidia/groupvit-gcc-yfcc](https://huggingface.co/nvidia/groupvit-gcc-yfcc) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`GroupViTTextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`GroupViTVisionConfig`].
        projection_dim (`int`, *optional*, defaults to 256):
            Dimentionality of text and vision projection layers.
        projection_intermediate_dim (`int`, *optional*, defaults to 4096):
            Dimentionality of intermediate layer of text and vision projection layers.
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            The inital value of the *logit_scale* parameter. Default is used as per the original GroupViT
            implementation.
        kwargs (*optional*):
            Dictionary of keyword arguments.
    """
    model_type = 'groupvit'

    def __init__(self, text_config=None, vision_config=None, projection_dim=256, projection_intermediate_dim=4096, logit_scale_init_value=2.6592, **kwargs):
        text_config_dict = kwargs.pop('text_config_dict', None)
        vision_config_dict = kwargs.pop('vision_config_dict', None)
        super().__init__(**kwargs)
        if text_config_dict is not None:
            if text_config is None:
                text_config = {}
            _text_config_dict = GroupViTTextConfig(**text_config_dict).to_dict()
            for key, value in _text_config_dict.items():
                if key in text_config and value != text_config[key] and (key not in ['transformers_version']):
                    if key in text_config_dict:
                        message = f'`{key}` is found in both `text_config_dict` and `text_config` but with different values. The value `text_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`text_config_dict` is provided which will be used to initialize `GroupViTTextConfig`. The value `text_config["{key}"]` will be overriden.'
                    logger.info(message)
            text_config.update(_text_config_dict)
        if vision_config_dict is not None:
            if vision_config is None:
                vision_config = {}
            _vision_config_dict = GroupViTVisionConfig(**vision_config_dict).to_dict()
            if 'id2label' in _vision_config_dict:
                _vision_config_dict['id2label'] = {str(key): value for key, value in _vision_config_dict['id2label'].items()}
            for key, value in _vision_config_dict.items():
                if key in vision_config and value != vision_config[key] and (key not in ['transformers_version']):
                    if key in vision_config_dict:
                        message = f'`{key}` is found in both `vision_config_dict` and `vision_config` but with different values. The value `vision_config_dict["{key}"]` will be used instead.'
                    else:
                        message = f'`vision_config_dict` is provided which will be used to initialize `GroupViTVisionConfig`. The value `vision_config["{key}"]` will be overriden.'
                    logger.info(message)
            vision_config.update(_vision_config_dict)
        if text_config is None:
            text_config = {}
            logger.info('`text_config` is `None`. Initializing the `GroupViTTextConfig` with default values.')
        if vision_config is None:
            vision_config = {}
            logger.info('`vision_config` is `None`. initializing the `GroupViTVisionConfig` with default values.')
        self.text_config = GroupViTTextConfig(**text_config)
        self.vision_config = GroupViTVisionConfig(**vision_config)
        self.projection_dim = projection_dim
        self.projection_intermediate_dim = projection_intermediate_dim
        self.logit_scale_init_value = logit_scale_init_value
        self.initializer_range = 0.02
        self.initializer_factor = 1.0
        self.output_segmentation = False

    @classmethod
    def from_text_vision_configs(cls, text_config: GroupViTTextConfig, vision_config: GroupViTVisionConfig, **kwargs):
        """
        Instantiate a [`GroupViTConfig`] (or a derived class) from groupvit text model configuration and groupvit
        vision model configuration.

        Returns:
            [`GroupViTConfig`]: An instance of a configuration object
        """
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)