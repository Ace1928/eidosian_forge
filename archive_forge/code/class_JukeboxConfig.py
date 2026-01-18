import os
from typing import List, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class JukeboxConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`JukeboxModel`].

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information. Instantiating a configuration with the defaults will
    yield a similar configuration to that of
    [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox-1b-lyrics) architecture.


    The downsampling and stride are used to determine downsampling of the input sequence. For example, downsampling =
    (5,3), and strides = (2, 2) will downsample the audio by 2^5 = 32 to get the first level of codes, and 2**8 = 256
    to get the second level codes. This is mostly true for training the top level prior and the upsamplers.

    Args:
        vqvae_config (`JukeboxVQVAEConfig`, *optional*):
            Configuration for the `JukeboxVQVAE` model.
        prior_config_list (`List[JukeboxPriorConfig]`, *optional*):
            List of the configs for each of the `JukeboxPrior` of the model. The original architecture uses 3 priors.
        nb_priors (`int`, *optional*, defaults to 3):
            Number of prior models that will sequentially sample tokens. Each prior is conditional auto regressive
            (decoder) model, apart from the top prior, which can include a lyric encoder. The available models were
            trained using a top prior and 2 upsampler priors.
        sampling_rate (`int`, *optional*, defaults to 44100):
            Sampling rate of the raw audio.
        timing_dims (`int`, *optional*, defaults to 64):
            Dimensions of the JukeboxRangeEmbedding layer which is equivalent to traditional positional embedding
            layer. The timing embedding layer converts the absolute and relative position in the currently sampled
            audio to a tensor of length `timing_dims` that will be added to the music tokens.
        min_duration (`int`, *optional*, defaults to 0):
            Minimum duration of the audios to generate
        max_duration (`float`, *optional*, defaults to 600.0):
            Maximum duration of the audios to generate
        max_nb_genres (`int`, *optional*, defaults to 5):
            Maximum number of genres that can be used to condition a single sample.
        metadata_conditioning (`bool`, *optional*, defaults to `True`):
            Whether or not to use metadata conditioning, corresponding to the artist, the genre and the min/maximum
            duration.

    Example:

    ```python
    >>> from transformers import JukeboxModel, JukeboxConfig

    >>> # Initializing a Jukebox configuration
    >>> configuration = JukeboxConfig()

    >>> # Initializing a model from the configuration
    >>> model = JukeboxModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    model_type = 'jukebox'

    def __init__(self, vqvae_config=None, prior_config_list=None, nb_priors=3, sampling_rate=44100, timing_dims=64, min_duration=0, max_duration=600.0, max_nb_genres=5, metadata_conditioning=True, **kwargs):
        if vqvae_config is None:
            vqvae_config = {}
            logger.info('vqvae_config is None. initializing the JukeboxVQVAE with default values.')
        self.vqvae_config = JukeboxVQVAEConfig(**vqvae_config)
        if prior_config_list is not None:
            self.prior_configs = [JukeboxPriorConfig(**prior_config) for prior_config in prior_config_list]
        else:
            self.prior_configs = []
            for prior_idx in range(nb_priors):
                prior_config = kwargs.pop(f'prior_{prior_idx}', None)
                if prior_config is None:
                    prior_config = {}
                    logger.info(f"prior_{prior_idx}'s  config is None. Initializing the JukeboxPriorConfig list with default values.")
                self.prior_configs.append(JukeboxPriorConfig(**prior_config))
        self.hop_fraction = self.vqvae_config.hop_fraction
        self.nb_priors = nb_priors
        self.max_nb_genres = max_nb_genres
        self.sampling_rate = sampling_rate
        self.timing_dims = timing_dims
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.metadata_conditioning = metadata_conditioning
        super().__init__(**kwargs)

    @classmethod
    def from_configs(cls, prior_configs: List[JukeboxPriorConfig], vqvae_config: JukeboxVQVAEConfig, **kwargs):
        """
        Instantiate a [`JukeboxConfig`] (or a derived class) from clip text model configuration and clip vision model
        configuration.

        Returns:
            [`JukeboxConfig`]: An instance of a configuration object
        """
        prior_config_list = [config.to_dict() for config in prior_configs]
        return cls(prior_config_list=prior_config_list, vqvae_config_dict=vqvae_config.to_dict(), **kwargs)

    def to_dict(self):
        result = super().to_dict()
        result['prior_config_list'] = [config.to_dict() for config in result.pop('prior_configs')]
        return result