import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...cache_utils import Cache, DynamicCache
from ...modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, _prepare_4d_causal_attention_mask_for_sdpa
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_starcoder2 import Starcoder2Config
@add_start_docstrings('The bare Starcoder2 Model outputting raw hidden-states without any specific head on top.', STARCODER2_START_DOCSTRING)
class Starcoder2PreTrainedModel(PreTrainedModel):
    config_class = Starcoder2Config
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['Starcoder2DecoderLayer']
    _skip_keys_device_placement = 'past_key_values'
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