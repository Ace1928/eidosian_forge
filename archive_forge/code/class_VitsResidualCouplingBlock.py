import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vits import VitsConfig
class VitsResidualCouplingBlock(nn.Module):

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.flows = nn.ModuleList()
        for _ in range(config.prior_encoder_num_flows):
            self.flows.append(VitsResidualCouplingLayer(config))

    def forward(self, inputs, padding_mask, global_conditioning=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                inputs, _ = flow(inputs, padding_mask, global_conditioning)
                inputs = torch.flip(inputs, [1])
        else:
            for flow in reversed(self.flows):
                inputs = torch.flip(inputs, [1])
                inputs, _ = flow(inputs, padding_mask, global_conditioning, reverse=True)
        return inputs