from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gemma.configuration_gemma import GemmaConfig
class UnpaddedGemmaRotaryEmbedding(torch.nn.Module):

    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        inv_freq = 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()
        self.register_buffer('cos_cached', emb.cos().to(dtype), persistent=False)
        self.register_buffer('sin_cached', emb.sin().to(dtype), persistent=False)

    def forward(self):
        return (self.cos_cached, self.sin_cached)