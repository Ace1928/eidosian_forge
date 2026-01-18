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
class VitsPosteriorEncoder(nn.Module):

    def __init__(self, config: VitsConfig):
        super().__init__()
        self.out_channels = config.flow_size
        self.conv_pre = nn.Conv1d(config.spectrogram_bins, config.hidden_size, 1)
        self.wavenet = VitsWaveNet(config, num_layers=config.posterior_encoder_num_wavenet_layers)
        self.conv_proj = nn.Conv1d(config.hidden_size, self.out_channels * 2, 1)

    def forward(self, inputs, padding_mask, global_conditioning=None):
        inputs = self.conv_pre(inputs) * padding_mask
        inputs = self.wavenet(inputs, padding_mask, global_conditioning)
        stats = self.conv_proj(inputs) * padding_mask
        mean, log_stddev = torch.split(stats, self.out_channels, dim=1)
        sampled = (mean + torch.randn_like(mean) * torch.exp(log_stddev)) * padding_mask
        return (sampled, mean, log_stddev)