import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput
from ...time_series_utils import NegativeBinomialOutput, NormalOutput, StudentTOutput
from ...utils import (
from .configuration_patchtsmixer import PatchTSMixerConfig
class PatchTSMixerForPredictionHead(nn.Module):
    """Prediction Head for Forecasting

    Args:
        config (`PatchTSMixerConfig`, *required*): Configuration.
    """

    def __init__(self, config: PatchTSMixerConfig, distribution_output=None):
        super().__init__()
        self.prediction_channel_indices = config.prediction_channel_indices
        if self.prediction_channel_indices is not None:
            self.prediction_channel_indices.sort()
        self.dropout_layer = nn.Dropout(config.head_dropout)
        if distribution_output is None:
            self.base_forecast_block = nn.Linear(config.num_patches * config.d_model, config.prediction_length)
        else:
            self.base_forecast_block = distribution_output.get_parameter_projection(config.num_patches * config.d_model)
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, hidden_features):
        """

        Args:
            hidden_features (`torch.Tensor` of shape `(batch_size, num_patch, d_model)` in `flatten` mode
                or `(batch_size, n_vars, num_patch, d_model)` in `common_channel`/`mix_channel` mode.): Input hidden
                features.

        Returns:
            `torch.Tensor` of shape `(batch_size, prediction_length, nvars)`.

        """
        hidden_features = self.flatten(hidden_features)
        hidden_features = self.dropout_layer(hidden_features)
        forecast = self.base_forecast_block(hidden_features)
        if isinstance(forecast, tuple):
            forecast = tuple((z.transpose(-1, -2) for z in forecast))
        else:
            forecast = forecast.transpose(-1, -2)
        if self.prediction_channel_indices is not None:
            if isinstance(forecast, tuple):
                forecast = tuple((z[..., self.prediction_channel_indices] for z in forecast))
            else:
                forecast = forecast[..., self.prediction_channel_indices]
        return forecast