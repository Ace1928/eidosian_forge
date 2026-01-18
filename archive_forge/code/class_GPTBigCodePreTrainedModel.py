import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_gpt_bigcode import GPTBigCodeConfig
class GPTBigCodePreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = GPTBigCodeConfig
    base_model_prefix = 'transformer'
    supports_gradient_checkpointing = True
    _no_split_modules = ['GPTBigCodeBlock']
    _skip_keys_device_placement = 'past_key_values'
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (GPTBigCodeMLP, GPTBigCodeAttention)):
            module.c_proj.weight.data.normal_(mean=0.0, std=self.config.initializer_range / math.sqrt(2 * self.config.n_layer))
            module.c_proj._is_hf_initialized = True
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)