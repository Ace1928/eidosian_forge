import math
import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_llama import LlamaConfig
def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int]=None):
    if self.config._attn_implementation == 'flash_attention_2' and cache_cls == StaticCache:
        raise ValueError('`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers')
    if max_cache_len > self.model.causal_mask.shape[-1] or self.device != self.model.causal_mask.device:
        causal_mask = torch.full((max_cache_len, max_cache_len), fill_value=True, device=self.device, dtype=torch.bool)
        self.register_buffer('causal_mask', torch.triu(causal_mask, diagonal=1), persistent=False)
    for layer in self.model.layers:
        weights = layer.self_attn.o_proj.weight
        layer.self_attn.past_key_value = cache_cls(self.config, max_batch_size, max_cache_len, device=weights.device, dtype=weights.dtype)