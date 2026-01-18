import os
from typing import List, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class JukeboxVQVAEConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`JukeboxVQVAE`]. It is used to instantiate a
    `JukeboxVQVAE` according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VQVAE from
    [openai/jukebox-1b-lyrics](https://huggingface.co/openai/jukebox-1b-lyrics) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        act_fn (`str`, *optional*, defaults to `"relu"`):
            Activation function of the model.
        nb_discrete_codes (`int`, *optional*, defaults to 2048):
            Number of codes of the VQVAE.
        commit (`float`, *optional*, defaults to 0.02):
            Commit loss multiplier.
        conv_input_shape (`int`, *optional*, defaults to 1):
            Number of audio channels.
        conv_res_scale (`bool`, *optional*, defaults to `False`):
            Whether or not to scale the residuals of the `JukeboxResConv1DBlock`.
        embed_dim (`int`, *optional*, defaults to 64):
            Embedding dimension of the codebook vectors.
        hop_fraction (`List[int]`, *optional*, defaults to `[0.125, 0.5, 0.5]`):
            Fraction of non-intersecting window used when continuing the sampling process.
        levels (`int`, *optional*, defaults to 3):
            Number of hierarchical levels that used in the VQVAE.
        lmu (`float`, *optional*, defaults to 0.99):
            Used in the codebook update, exponential moving average coefficient. For more detail refer to Appendix A.1
            of the original [VQVAE paper](https://arxiv.org/pdf/1711.00937v2.pdf)
        multipliers (`List[int]`, *optional*, defaults to `[2, 1, 1]`):
            Depth and width multipliers used for each level. Used on the `res_conv_width` and `res_conv_depth`
        res_conv_depth (`int`, *optional*, defaults to 4):
            Depth of the encoder and decoder block. If no `multipliers` are used, this is the same for each level.
        res_conv_width (`int`, *optional*, defaults to 32):
            Width of the encoder and decoder block. If no `multipliers` are used, this is the same for each level.
        res_convolution_multiplier (`int`, *optional*, defaults to 1):
            Scaling factor of the hidden dimension used in the `JukeboxResConv1DBlock`.
        res_dilation_cycle (`int`, *optional*):
            Dilation cycle value used in the `JukeboxResnet`. If an int is used, each new Conv1 block will have a depth
            reduced by a power of `res_dilation_cycle`.
        res_dilation_growth_rate (`int`, *optional*, defaults to 3):
            Resnet dilation growth rate used in the VQVAE (dilation_growth_rate ** depth)
        res_downs_t (`List[int]`, *optional*, defaults to `[3, 2, 2]`):
            Downsampling rate for each level of the hierarchical VQ-VAE.
        res_strides_t (`List[int]`, *optional*, defaults to `[2, 2, 2]`):
            Stride used for each level of the hierarchical VQ-VAE.
        sample_length (`int`, *optional*, defaults to 1058304):
            Provides the max input shape of the VQVAE. Is used to compute the input shape of each level.
        init_scale (`float`, *optional*, defaults to 0.2):
            Initialization scale.
        zero_out (`bool`, *optional*, defaults to `False`):
            Whether or not to zero out convolution weights when initializing.
    """
    model_type = 'jukebox_vqvae'

    def __init__(self, act_fn='relu', nb_discrete_codes=2048, commit=0.02, conv_input_shape=1, conv_res_scale=False, embed_dim=64, hop_fraction=[0.125, 0.5, 0.5], levels=3, lmu=0.99, multipliers=[2, 1, 1], res_conv_depth=4, res_conv_width=32, res_convolution_multiplier=1, res_dilation_cycle=None, res_dilation_growth_rate=3, res_downs_t=[3, 2, 2], res_strides_t=[2, 2, 2], sample_length=1058304, init_scale=0.2, zero_out=False, **kwargs):
        self.hop_fraction = hop_fraction
        self.conv_input_shape = conv_input_shape
        self.sample_length = sample_length
        self.levels = levels
        self.embed_dim = embed_dim
        self.nb_discrete_codes = nb_discrete_codes
        self.res_conv_width = res_conv_width
        self.res_conv_depth = res_conv_depth
        self.res_convolution_multiplier = res_convolution_multiplier
        self.res_dilation_growth_rate = res_dilation_growth_rate
        self.res_dilation_cycle = res_dilation_cycle
        self.multipliers = multipliers
        self.res_downs_t = res_downs_t
        self.res_strides_t = res_strides_t
        self.lmu = lmu
        self.commit = commit
        self.conv_res_scale = conv_res_scale
        self.act_fn = act_fn
        self.init_scale = init_scale
        self.zero_out = zero_out

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'jukebox':
            config_dict = config_dict['vqvae_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)