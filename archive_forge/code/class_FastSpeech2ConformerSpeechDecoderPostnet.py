import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
class FastSpeech2ConformerSpeechDecoderPostnet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.feat_out = nn.Linear(config.hidden_size, config.num_mel_bins * config.reduction_factor)
        self.layers = nn.ModuleList([FastSpeech2ConformerBatchNormConvLayer(config, i) for i in range(config.speech_decoder_postnet_layers)])

    def forward(self, hidden_states: torch.Tensor):
        outputs_before_postnet = self.feat_out(hidden_states).view(hidden_states.size(0), -1, self.config.num_mel_bins)
        layer_output = outputs_before_postnet.transpose(1, 2)
        for layer in self.layers:
            layer_output = layer(layer_output)
        outputs_after_postnet = outputs_before_postnet + layer_output.transpose(1, 2)
        return (outputs_before_postnet, outputs_after_postnet)