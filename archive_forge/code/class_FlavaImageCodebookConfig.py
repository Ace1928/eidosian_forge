import os
from typing import Any, Dict, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class FlavaImageCodebookConfig(PretrainedConfig):
    model_type = 'flava_image_codebook'
    "\n    [`FlavaImageCodebookConfig`] is the configuration class to store the configuration of a [`FlavaImageCodebook`]. It\n    is used to instantiate an FLAVA model according to the specified arguments, defining the model architecture.\n    Instantiating a configuration with the defaults will yield a similar configuration to that of the FLAVA\n    [facebook/flava-image-codebook](https://huggingface.co/facebook/flava-image-codebook) architecture.\n\n    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the\n    documentation from [`PretrainedConfig`] for more information.\n\n    Args:\n        num_groups (`int`, defaults to 4):\n            Number of groups to be created. This parameter as of now doesn't affect the model and is used for some\n            internal calculation and estimations.\n        input_channels (`int`, defaults to 3):\n            Number of channels in the image to be passed.\n        num_blocks_per_group (`int`, defaults to 2):\n            Number of conv-based blocks per group.\n        hidden_size (`int`, defaults to 256):\n            Size of hidden dim for the blocks.\n        vocab_size (`int`, defaults to 8192):\n            Size of the output vocabulary for the codebook.\n        freeze (`bool`, defaults to `True`):\n            Whether to freeze the weights of the model.\n        initializer_range (`float`, *optional*, defaults to 0.02):\n            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.\n        kwargs (*optional*):\n            Dictionary of keyword arguments.\n\n    Example:\n\n    ```python\n    >>> from transformers import FlavaImageCodebookConfig, FlavaImageCodebook\n\n    >>> # Initializing a FlavaImageCodebook with style configuration\n    >>> configuration = FlavaImageCodebookConfig()\n\n    >>> # Initializing a FlavaImageCodebook model (with random weights) from the style configuration\n    >>> model = FlavaImageCodebook(configuration)\n    >>> # Accessing the model configuration\n    >>> configuration = model.config\n    ```\n    "

    def __init__(self, num_groups: int=4, input_channels: int=3, num_blocks_per_group: int=2, hidden_size: int=256, vocab_size: int=8192, freeze: int=True, initializer_range: float=0.02, **kwargs):
        super().__init__(**kwargs)
        self.num_groups = num_groups
        self.input_channels = input_channels
        self.num_blocks_per_group = num_blocks_per_group
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.freeze = freeze
        self.initializer_range = initializer_range

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'flava':
            config_dict = config_dict['image_codebook_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)