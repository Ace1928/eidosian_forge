import math
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache, StaticCache
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_dbrx import DbrxConfig
@add_start_docstrings('The bare DBRX Model outputting raw hidden-states without any specific head on top.', DBRX_START_DOCSTRING)
class DbrxPreTrainedModel(PreTrainedModel):
    config_class = DbrxConfig
    base_model_prefix = 'transformer'
    supports_gradient_checkpointing = True
    _no_split_modules = ['DbrxBlock']
    _skip_keys_device_placement = ['past_key_values']
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, DbrxExpertGLU):
            module.w1.data.normal_(mean=0.0, std=std)
            module.v1.data.normal_(mean=0.0, std=std)
            module.w2.data.normal_(mean=0.0, std=std)

    def _setup_cache(self, cache_cls: Any, max_batch_size: int, max_cache_len: int):
        if self.config._attn_implementation == 'flash_attention_2' and cache_cls == StaticCache:
            raise ValueError('`static` cache implementation is not compatible with ' + '`attn_implementation==flash_attention_2`. Make sure to use ' + '`spda` in the mean time and open an issue at https://github.com/huggingface/transformers.')
        for block in self.transformer.blocks:
            device = block.norm_attn_norm.norm_1.weight.device
            if hasattr(self.config, '_pre_quantization_dtype'):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = block.norm_attn_norm.attn.out_proj.weight.dtype
            block.norm_attn_norm.attn.past_key_value = cache_cls(self.config, max_batch_size, max_cache_len, device=device, dtype=dtype)

    def _reset_cache(self):
        for block in self.transformer.blocks:
            block.norm_attn_norm.attn.past_key_value = None