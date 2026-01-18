from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
from .configuration_time_series_transformer import TimeSeriesTransformerConfig
class TimeSeriesMeanScaler(nn.Module):
    """
    Computes a scaling factor as the weighted average absolute value along the first dimension, and scales the data
    accordingly.
    """

    def __init__(self, config: TimeSeriesTransformerConfig):
        super().__init__()
        self.dim = config.scaling_dim if hasattr(config, 'scaling_dim') else 1
        self.keepdim = config.keepdim if hasattr(config, 'keepdim') else True
        self.minimum_scale = config.minimum_scale if hasattr(config, 'minimum_scale') else 1e-10
        self.default_scale = config.default_scale if hasattr(config, 'default_scale') else None

    def forward(self, data: torch.Tensor, observed_indicator: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
            data (`torch.Tensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                input for Batch norm calculation
            observed_indicator (`torch.BoolTensor` of shape `(batch_size, sequence_length, num_input_channels)`):
                Calculating the scale on the observed indicator.
        Returns:
            tuple of `torch.Tensor` of shapes
                (`(batch_size, sequence_length, num_input_channels)`,`(batch_size, 1, num_input_channels)`,
                `(batch_size, 1, num_input_channels)`)
        """
        ts_sum = (data * observed_indicator).abs().sum(self.dim, keepdim=True)
        num_observed = observed_indicator.sum(self.dim, keepdim=True)
        scale = ts_sum / torch.clamp(num_observed, min=1)
        if self.default_scale is None:
            batch_sum = ts_sum.sum(dim=0)
            batch_observations = torch.clamp(num_observed.sum(0), min=1)
            default_scale = torch.squeeze(batch_sum / batch_observations)
        else:
            default_scale = self.default_scale * torch.ones_like(scale)
        scale = torch.where(num_observed > 0, scale, default_scale)
        scale = torch.clamp(scale, min=self.minimum_scale)
        scaled_data = data / scale
        if not self.keepdim:
            scale = scale.squeeze(dim=self.dim)
        return (scaled_data, torch.zeros_like(scale), scale)