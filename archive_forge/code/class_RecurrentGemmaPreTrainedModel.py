import math
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import AttentionMaskConverter
from ...modeling_outputs import BaseModelOutputWithNoAttention, CausalLMOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
from ...utils import (
from .configuration_recurrent_gemma import RecurrentGemmaConfig
@add_start_docstrings('The bare RecurrentGemma Model outputting raw hidden-states without any specific head on top.', RECURRENTGEMMA_START_DOCSTRING)
class RecurrentGemmaPreTrainedModel(PreTrainedModel):
    config_class = RecurrentGemmaConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['RecurrentGemmaDecoderLayer']
    _skip_keys_device_placement = ['cache']
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = math.sqrt(self.config.w_init_variance_scale / self.config.conv1d_width)
        if isinstance(module, nn.Conv1d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            torch.nn.init.zeros_(module.bias)
        elif isinstance(module, RecurrentGemmaSdpaAttention):
            torch.nn.init.normal_(module.q_proj.weight, mean=0.0, std=math.sqrt(1.0 / self.config.hidden_size))
            torch.nn.init.normal_(module.k_proj.weight, mean=0.0, std=math.sqrt(1.0 / self.config.hidden_size))
            torch.nn.init.normal_(module.v_proj.weight, mean=0.0, std=math.sqrt(1.0 / self.config.hidden_size))
            std = math.sqrt(self.config.final_w_init_variance_scale / self.config.hidden_size)
            torch.nn.init.normal_(module.o_proj.weight, mean=0.0, std=std)
        elif isinstance(module, RecurrentGemmaRecurrentBlock):
            torch.nn.init.zeros_(module.linear_x.bias)
            torch.nn.init.normal_(module.linear_x.weight, mean=0.0, std=math.sqrt(1.0 / self.config.hidden_size))
            torch.nn.init.zeros_(module.linear_y.bias)
            torch.nn.init.normal_(module.linear_y.weight, mean=0.0, std=math.sqrt(1.0 / self.config.hidden_size))
            std = math.sqrt(self.config.final_w_init_variance_scale / self.config.lru_width)
            torch.nn.init.normal_(module.linear_out.weight, mean=0.0, std=std)
            torch.nn.init.zeros_(module.linear_out.bias)
        elif isinstance(module, RecurrentGemmaRglru):
            std = math.sqrt(self.config.w_init_variance_scale / (self.config.lru_width // self.config.num_attention_heads))
            torch.nn.init.normal_(module.input_gate_weight, mean=0.0, std=std)
            torch.nn.init.normal_(module.recurrent_gate_weight, mean=0.0, std=std)
            torch.nn.init.zeros_(module.input_gate_bias)
            torch.nn.init.zeros_(module.recurrent_gate_bias)
            module.recurrent_param.data.uniform_(0.9 ** 2 + 1e-08, 0.999 ** 2 + 1e-08)
            module.recurrent_param.data.log_().mul_(0.5)
            module.recurrent_param.data.neg_().exp_().sub_(1.0).log_()
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if getattr(module, 'bias', None) is not None:
                torch.nn.init.zeros_(module.bias)

    def _setup_cache(self, config, batch, device, dtype):
        layers = getattr(self, 'model', self).layers
        for layer in layers:
            layer.temporal_block._setup_cache(batch, device, dtype)

    def reset_cache(self, batch, device, dtype):
        pass