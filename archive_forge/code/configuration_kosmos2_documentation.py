import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging

    This is the configuration class to store the configuration of a [`Kosmos2Model`]. It is used to instantiate a
    KOSMOS-2 model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the KOSMOS-2
    [microsoft/kosmos-2-patch14-224](https://huggingface.co/microsoft/kosmos-2-patch14-224) architecture.

    Args:
        text_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2TextConfig`].
        vision_config (`dict`, *optional*):
            Dictionary of configuration options used to initialize [`Kosmos2VisionConfig`].
        latent_query_num (`int`, *optional*, defaults to 64):
            The number of latent query tokens that represent the image features used in the text decoder component.
        kwargs (*optional*):
            Dictionary of keyword arguments.

    Example:

    ```python
    >>> from transformers import Kosmos2Config, Kosmos2Model

    >>> # Initializing a Kosmos-2 kosmos-2-patch14-224 style configuration
    >>> configuration = Kosmos2Config()

    >>> # Initializing a model (with random weights) from the kosmos-2-patch14-224 style configuration
    >>> model = Kosmos2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```