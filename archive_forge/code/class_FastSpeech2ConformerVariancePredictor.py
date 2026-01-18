import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
class FastSpeech2ConformerVariancePredictor(nn.Module):

    def __init__(self, config: FastSpeech2ConformerConfig, num_layers=2, num_chans=384, kernel_size=3, dropout_rate=0.5):
        """
        Initilize variance predictor module.

        Args:
            input_dim (`int`): Input dimension.
            num_layers (`int`, *optional*, defaults to 2): Number of convolutional layers.
            num_chans (`int`, *optional*, defaults to 384): Number of channels of convolutional layers.
            kernel_size (`int`, *optional*, defaults to 3): Kernel size of convolutional layers.
            dropout_rate (`float`, *optional*, defaults to 0.5): Dropout rate.
        """
        super().__init__()
        self.conv_layers = nn.ModuleList()
        for idx in range(num_layers):
            input_channels = config.hidden_size if idx == 0 else num_chans
            layer = FastSpeech2ConformerPredictorLayer(input_channels, num_chans, kernel_size, dropout_rate)
            self.conv_layers.append(layer)
        self.linear = nn.Linear(num_chans, 1)

    def forward(self, encoder_hidden_states, padding_masks=None):
        """
        Calculate forward propagation.

        Args:
            encoder_hidden_states (`torch.Tensor` of shape `(batch_size, max_text_length, input_dim)`):
                Batch of input sequences.
            padding_masks (`torch.ByteTensor` of shape `(batch_size, max_text_length)`, *optional*):
                Batch of masks indicating padded part.

        Returns:
            Tensor: Batch of predicted sequences `(batch_size, max_text_length, 1)`.
        """
        hidden_states = encoder_hidden_states.transpose(1, -1)
        for layer in self.conv_layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.linear(hidden_states.transpose(1, 2))
        if padding_masks is not None:
            hidden_states = hidden_states.masked_fill(padding_masks, 0.0)
        return hidden_states