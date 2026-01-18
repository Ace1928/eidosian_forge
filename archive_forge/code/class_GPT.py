import math
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
from torch.nn import functional as F
from ray.rllib.utils.annotations import DeveloperAPI
from ray.rllib.utils.deprecation import Deprecated
@Deprecated(error=False)
class GPT(nn.Module):
    """GPT Transformer Model"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.block_size is not None
        self.block_size = config.block_size
        self.transformer = nn.ModuleDict(dict(drop=nn.Dropout(config.embed_pdrop), h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]), ln_f=nn.LayerNorm(config.n_embed)))
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, input_embeds, attention_masks=None, return_attentions=False):
        """
        input_embeds: [batch_size x seq_len x n_embed]
        attention_masks: [batch_size x seq_len], 0 don't attend, 1 attend
        """
        B, T, C = input_embeds.size()
        assert T <= self.block_size, f'Cannot forward sequence of length {T}, block size is only {self.block_size}'
        if attention_masks is not None:
            _B, _T = attention_masks.size()
            assert _B == B and _T == T
            attention_masks = attention_masks[:, None, None, :]
            attention_masks = attention_masks.to(dtype=input_embeds.dtype)
            attention_masks = (1.0 - attention_masks) * -1000000000.0
        x = self.transformer.drop(input_embeds)
        atts = []
        for block in self.transformer.h:
            x, att = block(x, attention_masks=attention_masks)
            atts.append(att)
        x = self.transformer.ln_f(x)
        if return_attentions:
            return (x, atts)
        else:
            return x