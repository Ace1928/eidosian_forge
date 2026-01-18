from typing import TYPE_CHECKING, Optional, Union
from ...configuration_utils import PretrainedConfig
from ...utils import logging
class MptAttentionConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`MptAttention`] class. It is used to instantiate
    attention layers according to the specified arguments, defining the layers architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MPT
    [mosaicml/mpt-7b](https://huggingface.co/mosaicml/mpt-7b) architecture. Most of the arguments are kept for backward
    compatibility with previous MPT models that are hosted on the Hub (previously with `trust_remote_code=True`).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        attn_type (`str`, *optional*, defaults to `"multihead_attention"`):
            type of attention to use. Options: `"multihead_attention"`, `"multiquery_attention"`.
        attn_pdrop (`float`, *optional*, defaults to 0.0):
            The dropout probability for the attention layers.
        attn_impl (`str`, *optional*, defaults to `"torch"`):
            The attention implementation to use. One of `"torch"`, `"flash"`, or `"triton"`.
        clip_qkv (`float`, *optional*):
            If not `None`, clip the queries, keys, and values in the attention layer to this value.
        softmax_scale (`float`, *optional*, defaults to `None`):
            If not `None`, scale the softmax in the attention layer by this value. If `None`, will default to
            `1/sqrt(hidden_size)`.
        prefix_lm (`bool`, *optional*, defaults to `False`)):
            Whether the model should operate as a Prefix LM. This requires passing an extra `prefix_mask` argument
            which indicates which tokens belong to the prefix. Tokens in the prefix can attend to one another
            bi-directionally. Tokens outside the prefix use causal attention.
        qk_ln (`bool`, *optional*, defaults to `False`):
            Whether to apply layer normalization to the queries and keys in the attention layer.
        attn_uses_sequence_id (`bool`, *optional*, defaults to `False`)):
            Whether to restrict attention to tokens that have the same token_type_ids. When the model is in `train`
            mode, this requires passing an extra *token_type_ids* argument which indicates which sub-sequence each
            token belongs to. Defaults to `False` meaning any provided *token_type_ids* will be ignored.
        alibi (`bool`, *optional*, defaults to `True`):
            Whether or not to use the alibi bias instead of positional embedding.
        alibi_bias_max (`int`, *optional*, defaults to 8):
            The maximum value of the alibi bias.
    """

    def __init__(self, attn_type='multihead_attention', attn_pdrop=0, attn_impl='torch', clip_qkv=None, softmax_scale=None, prefix_lm=False, qk_ln=False, attn_uses_sequence_id=False, alibi=True, alibi_bias_max=8, **kwargs):
        super().__init__()
        self.attn_type = attn_type
        self.attn_pdrop = attn_pdrop
        self.attn_impl = attn_impl
        self.clip_qkv = clip_qkv
        self.softmax_scale = softmax_scale
        self.prefix_lm = prefix_lm
        self.attn_uses_sequence_id = attn_uses_sequence_id
        self.alibi = alibi
        self.qk_ln = qk_ln
        self.alibi_bias_max = alibi_bias_max
        if attn_type not in ['multihead_attention', 'multiquery_attention']:
            raise ValueError(f'`attn_type` has to be either `multihead_attention` or `multiquery_attention`. Received: {attn_type}')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs) -> 'PretrainedConfig':
        cls._set_token_in_kwargs(kwargs)
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if config_dict.get('model_type') == 'mpt':
            config_dict = config_dict['attn_config']
        if 'model_type' in config_dict and hasattr(cls, 'model_type') and (config_dict['model_type'] != cls.model_type):
            logger.warning(f'You are using a model of type {config_dict['model_type']} to instantiate a model of type {cls.model_type}. This is not supported for all configurations of models and can yield errors.')
        return cls.from_dict(config_dict, **kwargs)