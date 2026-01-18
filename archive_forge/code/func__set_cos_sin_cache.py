import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ....activations import ACT2FN
from ....modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from ....modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ....modeling_utils import PreTrainedModel
from ....utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_open_llama import OpenLlamaConfig
def _set_cos_sin_cache(self, seq_len, device, dtype):
    self.max_seq_len_cached = seq_len
    if seq_len > self.max_position_embeddings:
        base = self.base * (self.scaling_factor * seq_len / self.max_position_embeddings - (self.scaling_factor - 1)) ** (self.dim / (self.dim - 2))
        inv_freq = 1.0 / base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
    t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
    freqs = torch.outer(t, self.inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
    self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)