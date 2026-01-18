from typing import Dict
from ...configuration_utils import PretrainedConfig
from ...utils import logging

    This is the configuration class to store the configuration of a [`FastSpeech2ConformerWithHifiGan`]. It is used to
    instantiate a `FastSpeech2ConformerWithHifiGanModel` model according to the specified sub-models configurations,
    defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the
    FastSpeech2ConformerModel [espnet/fastspeech2_conformer](https://huggingface.co/espnet/fastspeech2_conformer) and
    FastSpeech2ConformerHifiGan
    [espnet/fastspeech2_conformer_hifigan](https://huggingface.co/espnet/fastspeech2_conformer_hifigan) architectures.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        model_config (`typing.Dict`, *optional*):
            Configuration of the text-to-speech model.
        vocoder_config (`typing.Dict`, *optional*):
            Configuration of the vocoder model.
    model_config ([`FastSpeech2ConformerConfig`], *optional*):
        Configuration of the text-to-speech model.
    vocoder_config ([`FastSpeech2ConformerHiFiGanConfig`], *optional*):
        Configuration of the vocoder model.

    Example:

    ```python
    >>> from transformers import (
    ...     FastSpeech2ConformerConfig,
    ...     FastSpeech2ConformerHifiGanConfig,
    ...     FastSpeech2ConformerWithHifiGanConfig,
    ...     FastSpeech2ConformerWithHifiGan,
    ... )

    >>> # Initializing FastSpeech2ConformerWithHifiGan sub-modules configurations.
    >>> model_config = FastSpeech2ConformerConfig()
    >>> vocoder_config = FastSpeech2ConformerHifiGanConfig()

    >>> # Initializing a FastSpeech2ConformerWithHifiGan module style configuration
    >>> configuration = FastSpeech2ConformerWithHifiGanConfig(model_config.to_dict(), vocoder_config.to_dict())

    >>> # Initializing a model (with random weights)
    >>> model = FastSpeech2ConformerWithHifiGan(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    