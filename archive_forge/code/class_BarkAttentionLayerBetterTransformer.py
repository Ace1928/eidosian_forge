from typing import TYPE_CHECKING
import torch
import torch.nn as nn
from transformers.models.bart.modeling_bart import BartAttention
from transformers.models.blenderbot.modeling_blenderbot import BlenderbotAttention
from transformers.models.bloom.modeling_bloom import BloomAttention
from transformers.models.codegen.modeling_codegen import CodeGenAttention
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoSelfAttention
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXAttention
from transformers.models.gptj.modeling_gptj import GPTJAttention
from transformers.models.m2m_100.modeling_m2m_100 import M2M100Attention
from transformers.models.marian.modeling_marian import MarianAttention
from transformers.models.opt.modeling_opt import OPTAttention
from transformers.models.pegasus.modeling_pegasus import PegasusAttention
from transformers.models.t5.modeling_t5 import T5Attention
from ...utils.import_utils import check_if_transformers_greater
from .attention import (
from .base import BetterTransformerBaseLayer
class BarkAttentionLayerBetterTransformer(BetterTransformerBaseLayer, BarkSelfAttention, nn.Module):
    _attn = bark_wrapped_scaled_dot_product

    def __init__(self, layer: 'nn.Module', config: 'PretrainedConfig', is_causal: bool=False):
        super().__init__(config)
        is_causal = layer.is_causal
        config.dropout = layer.dropout
        config.hidden_size = layer.embed_dim
        config.num_heads = layer.num_heads
        config.bias = layer.out_proj.bias is not None
        if is_causal:
            config.block_size = layer.bias.shape[-1]
        with torch.device('meta'):
            super(BetterTransformerBaseLayer, self).__init__(config, is_causal)
        self.module_mapping = None
        submodules = ['dropout', 'attn_dropout', 'resid_dropout', 'att_proj', 'out_proj']
        for attr in submodules:
            setattr(self, attr, getattr(layer, attr))
        self.original_layers_mapping = {submodule: submodule for submodule in submodules}
        if is_causal:
            setattr(self, 'bias', getattr(layer, 'bias'))
            self.original_layers_mapping['bias'] = 'bias'
        self.supports_training = False
        self.dropout_prob_attn = float(config.dropout)

    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)