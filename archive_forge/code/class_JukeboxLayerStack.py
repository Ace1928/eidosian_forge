import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
class JukeboxLayerStack(nn.Module):

    def __init__(self, config, n_ctx):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = config.hidden_size
        self.num_layers = config.num_layers
        self.blocks = config.blocks
        self.attention_pattern = config.attention_pattern
        if self.blocks is not None:
            self.block_ctx = n_ctx // self.blocks
        self.encoder_len = config.nb_relevant_lyric_tokens
        self.n_heads = config.n_heads
        attention_pattern = ATTENTION_PATTERNS[self.attention_pattern]
        self._attn_mods = nn.ModuleList()
        for depth in range(self.num_layers):
            self._attn_mods.append(JukeboxBlock(config, n_ctx, attn_func=attention_pattern(depth)))
        self.saved_attn_weights = []

    def set_record_attn(self, record_attn):
        """
        Makes forward prop dump self-attention softmaxes to self.saved_attn_weights.

        Args:
            record_attn (`Union[bool,set]`):
                Either a set of layer indices indicating which layers to store, or a boolean value indicating Whether
                to dump all.
        """

        def _should_record_attn(layer_idx):
            if isinstance(record_attn, bool):
                return record_attn
            return layer_idx in record_attn
        for i, layer in enumerate(self._attn_mods):
            layer.attn.record_attn = _should_record_attn(i)
        if not record_attn:
            self.saved_attn_weights = []

    def forward(self, hidden_states, last_encoder_hidden_states=None, sample=False):
        for i, attn_layer in enumerate(self._attn_mods):
            if attn_layer.attn_func == 'cross_attention':
                hidden_states = attn_layer(hidden_states, last_encoder_hidden_states=last_encoder_hidden_states, sample=sample)
            else:
                hidden_states = attn_layer(hidden_states, last_encoder_hidden_states=None, sample=sample)
            if attn_layer.attn.record_attn:
                self.saved_attn_weights.append(attn_layer.attn.c_attn.weight)
        return hidden_states

    def del_cache(self):
        for attn_layer in self._attn_mods:
            attn_layer.attn.del_cache()