import copy
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...generation import GenerationConfig
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel, SequenceSummary
from ...pytorch_utils import Conv1D
from ...utils import (
from .configuration_clvp import (
class ClvpPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = ClvpConfig
    base_model_prefix = 'clvp'
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = 'past_key_values'

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
        elif isinstance(module, (nn.Linear, Conv1D, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, ClvpEncoderMLP):
            factor = self.config.initializer_factor
            in_proj_std = module.config.hidden_size ** (-0.5) * (2 * module.config.num_hidden_layers) ** (-0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** (-0.5) * factor
            nn.init.normal_(module.fc1.proj.weight if getattr(module.fc1, 'proj') else module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        elif isinstance(module, ClvpEncoder):
            config = self.config.text_config if hasattr(self.config, 'text_config') else self.config
            factor = config.initializer_factor
            module.projection.weight.data.normal_(mean=0.0, std=factor * config.hidden_size ** (-0.5))
        elif isinstance(module, ClvpConditioningEncoder):
            module.mel_conv.weight.data.normal_(mean=0.0, std=factor)
            module.mel_conv.bias.data.zero_()
        elif isinstance(module, ClvpForCausalLM):
            for name, p in module.named_parameters():
                if name == 'c_proj.weight':
                    p.data.normal_(mean=0.0, std=self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers))
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)