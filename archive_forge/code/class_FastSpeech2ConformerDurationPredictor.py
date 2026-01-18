import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
class FastSpeech2ConformerDurationPredictor(nn.Module):
    """
    Duration predictor module.

    This is a module of duration predictor described in the paper 'FastSpeech: Fast, Robust and Controllable Text to
    Speech' https://arxiv.org/pdf/1905.09263.pdf The duration predictor predicts a duration of each frame in log domain
    from the hidden embeddings of encoder.

    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`, the
        outputs are calculated in log domain but in `inference`, those are calculated in linear domain.

    """

    def __init__(self, config: FastSpeech2ConformerConfig):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.log_domain_offset = 1.0
        for layer_idx in range(config.duration_predictor_layers):
            num_chans = config.duration_predictor_channels
            input_channels = config.hidden_size if layer_idx == 0 else num_chans
            layer = FastSpeech2ConformerPredictorLayer(input_channels, num_chans, config.duration_predictor_kernel_size, config.duration_predictor_dropout_rate)
            self.conv_layers.append(layer)
        self.linear = nn.Linear(config.duration_predictor_channels, 1)

    def forward(self, encoder_hidden_states):
        """
        Args:
            hidden_states (`torch.Tensor` of shape `(batch_size, max_text_length, input_dim)`):
                Batch of input sequences.
            padding_masks (`torch.ByteTensor` of shape `(batch_size, max_text_length)`, *optional*):
                Batch of masks indicating padded part.

        Returns:
            `torch.Tensor`: Batch of predicted durations in log domain `(batch_size, max_text_length)`.

        """
        hidden_states = encoder_hidden_states.transpose(1, -1)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.linear(hidden_states.transpose(1, -1)).squeeze(-1)
        if not self.training:
            hidden_states = torch.clamp(torch.round(hidden_states.exp() - self.log_domain_offset), min=0).long()
        return hidden_states