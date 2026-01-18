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
@add_start_docstrings('The bare LLaMA Model outputting raw hidden-states without any specific head on top.', LLAMA_START_DOCSTRING)
class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['LlamaDecoderLayer']
    _skip_keys_device_placement = ['past_key_values', 'causal_mask']
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int]=None):
        if self.config._attn_implementation == 'flash_attention_2' and cache_cls == StaticCache:
            raise ValueError('`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers')
        if max_cache_len > self.model.causal_mask.shape[-1] or self.device != self.model.causal_mask.device:
            causal_mask = torch.full((max_cache_len, max_cache_len), fill_value=True, device=self.device, dtype=torch.bool)
            self.register_buffer('causal_mask', torch.triu(causal_mask, diagonal=1), persistent=False)
        for layer in self.model.layers:
            weights = layer.self_attn.o_proj.weight
            layer.self_attn.past_key_value = cache_cls(self.config, max_batch_size, max_cache_len, device=weights.device, dtype=weights.dtype)

    def _reset_cache(self):
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None