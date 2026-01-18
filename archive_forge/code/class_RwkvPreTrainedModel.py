import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_rwkv import RwkvConfig
class RwkvPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RwkvConfig
    base_model_prefix = 'rwkv'
    _no_split_modules = ['RwkvBlock']
    _keep_in_fp32_modules = ['time_decay', 'time_first']
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, RwkvSelfAttention):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            attention_hidden_size = module.attention_hidden_size
            ratio_0_to_1 = layer_id / (num_hidden_layers - 1)
            ratio_1_to_almost0 = 1.0 - layer_id / num_hidden_layers
            time_weight = torch.tensor([i / hidden_size for i in range(hidden_size)], dtype=module.time_mix_key.dtype, device=module.time_mix_key.device)
            time_weight = time_weight[None, None, :]
            decay_speed = [-5 + 8 * (h / (attention_hidden_size - 1)) ** (0.7 + 1.3 * ratio_0_to_1) for h in range(attention_hidden_size)]
            decay_speed = torch.tensor(decay_speed, dtype=module.time_decay.dtype, device=module.time_decay.device)
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(attention_hidden_size)], dtype=module.time_first.dtype, device=module.time_first.device) * 0.5
            with torch.no_grad():
                module.time_decay.data = decay_speed
                module.time_first.data = torch.ones_like(module.time_first * math.log(0.3) + zigzag)
                module.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_value.data = torch.pow(time_weight, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
                module.time_mix_receptance.data = torch.pow(time_weight, 0.5 * ratio_1_to_almost0)
        elif isinstance(module, RwkvFeedForward):
            layer_id = module.layer_id
            num_hidden_layers = module.config.num_hidden_layers
            hidden_size = module.config.hidden_size
            ratio_1_to_almost0 = 1.0 - layer_id / num_hidden_layers
            time_weight = torch.tensor([i / hidden_size for i in range(hidden_size)], dtype=module.time_mix_key.dtype, device=module.time_mix_key.device)
            time_weight = time_weight[None, None, :]
            with torch.no_grad():
                module.time_mix_key.data = torch.pow(time_weight, ratio_1_to_almost0)
                module.time_mix_receptance.data = torch.pow(time_weight, ratio_1_to_almost0)