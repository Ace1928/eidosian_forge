import os
from typing import Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
class Idefics2PerceiverConfig(PretrainedConfig):
    """
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the perceiver block.
        resampler_n_latents (`int`, *optional*, defaults to 64):
            Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
        resampler_depth (`int`, *optional*, defaults to 3):
            Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (<= 3).
        resampler_n_heads (`int`, *optional*, defaults to 16):
            Number of heads in each Transformer block (for multi-headed self-attention).
        resampler_head_dim (`int`, *optional*, defaults to 96):
            Dimensionality of each head projection in the Transformer block.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key-value heads in the perceiver attention block.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    """
    model_type = 'idefics2'

    def __init__(self, hidden_act='silu', resampler_n_latents=64, resampler_depth=3, resampler_n_heads=16, resampler_head_dim=96, num_key_value_heads=4, attention_dropout=0.0, **kwargs):
        self.hidden_act = hidden_act
        self.resampler_n_latents = resampler_n_latents
        self.resampler_depth = resampler_depth
        self.resampler_n_heads = resampler_n_heads
        self.num_key_value_heads = num_key_value_heads
        self.resampler_head_dim = resampler_head_dim
        self.attention_dropout = attention_dropout
        if self.num_key_value_heads > self.resampler_n_heads:
            raise ValueError(f'num_key_value_heads={self.num_key_value_heads} must be less than or equal to resampler_n_heads={self.resampler_n_heads}')
        super().__init__(**kwargs)