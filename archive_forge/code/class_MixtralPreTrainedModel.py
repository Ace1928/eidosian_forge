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
from ...modeling_attn_mask_utils import (
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
from ...utils.import_utils import is_torch_fx_available
from .configuration_mixtral import MixtralConfig
@add_start_docstrings('The bare Mixtral Model outputting raw hidden-states without any specific head on top.', MIXTRAL_START_DOCSTRING)
class MixtralPreTrainedModel(PreTrainedModel):
    config_class = MixtralConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['MixtralDecoderLayer']
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