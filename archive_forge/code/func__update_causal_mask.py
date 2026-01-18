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
def _update_causal_mask(self, attention_mask, input_tensor):
    if self.config._attn_implementation == 'flash_attention_2':
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    batch_size, seq_length = input_tensor.shape[:2]
    dtype = input_tensor.dtype
    device = input_tensor.device
    if seq_length > self.causal_mask.shape[-1]:
        causal_mask = torch.full((2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]), fill_value=1)
        self.register_buffer('causal_mask', torch.triu(causal_mask, diagonal=1), persistent=False)
    min_dtype = torch.finfo(dtype).min
    causal_mask = self.causal_mask[None, None, :, :].repeat(batch_size, 1, 1, 1).to(dtype) * min_dtype
    causal_mask = causal_mask.to(dtype=dtype, device=device)
    if attention_mask is not None and attention_mask.dim() == 2:
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[..., :mask_length].eq(0.0) * attention_mask[:, None, None, :].eq(0.0)
        causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(padding_mask, min_dtype)
    if self.config._attn_implementation == 'sdpa' and attention_mask is not None:
        is_tracing = torch.jit.is_tracing() or isinstance(input_tensor, torch.fx.Proxy) or (hasattr(torch, '_dynamo') and torch._dynamo.is_compiling())
        if not is_tracing and torch.any(attention_mask != 1):
            causal_mask = causal_mask.mul(~torch.all(causal_mask == min_dtype, dim=-1, keepdim=True)).to(dtype)
    return causal_mask