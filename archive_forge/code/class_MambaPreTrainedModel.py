import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.import_utils import is_causal_conv1d_available, is_mamba_ssm_available
from .configuration_mamba import MambaConfig
from ..deprecated._archive_maps import MAMBA_PRETRAINED_MODEL_ARCHIVE_LIST  # noqa: F401, E402
class MambaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = MambaConfig
    base_model_prefix = 'backbone'
    _no_split_modules = ['MambaBlock']
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, MambaMixer):
            module.A_log._no_weight_decay = True
            module.D._no_weight_decay = True
            dt_init_std = self.config.time_step_rank ** (-0.5) * self.config.time_step_scale
            if self.config.time_step_init_scheme == 'constant':
                nn.init.constant_(module.dt_proj.weight, dt_init_std)
            elif self.config.time_step_init_scheme == 'random':
                nn.init.uniform_(module.dt_proj.weight, -dt_init_std, dt_init_std)
            dt = torch.exp(torch.rand(self.config.intermediate_size) * (math.log(self.config.time_step_max) - math.log(self.config.time_step_min)) + math.log(self.config.time_step_min)).clamp(min=self.config.time_step_floor)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            with torch.no_grad():
                module.dt_proj.bias.copy_(inv_dt)
            module.dt_proj.bias._no_reinit = True
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                if not getattr(module.bias, '_no_reinit', False):
                    nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
        if self.config.rescale_prenorm_residual:
            for name, p in module.named_parameters():
                if name in ['out_proj.weight']:
                    nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                    with torch.no_grad():
                        p /= math.sqrt(self.config.num_layers)