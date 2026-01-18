import math
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, L1Loss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_speecht5 import SpeechT5Config, SpeechT5HifiGanConfig
class SpeechT5BatchNormConvLayer(nn.Module):

    def __init__(self, config, layer_id=0):
        super().__init__()
        if layer_id == 0:
            in_conv_dim = config.num_mel_bins
        else:
            in_conv_dim = config.speech_decoder_postnet_units
        if layer_id == config.speech_decoder_postnet_layers - 1:
            out_conv_dim = config.num_mel_bins
        else:
            out_conv_dim = config.speech_decoder_postnet_units
        self.conv = nn.Conv1d(in_conv_dim, out_conv_dim, kernel_size=config.speech_decoder_postnet_kernel, stride=1, padding=(config.speech_decoder_postnet_kernel - 1) // 2, bias=False)
        self.batch_norm = nn.BatchNorm1d(out_conv_dim)
        if layer_id < config.speech_decoder_postnet_layers - 1:
            self.activation = nn.Tanh()
        else:
            self.activation = None
        self.dropout = nn.Dropout(config.speech_decoder_postnet_dropout)

    def forward(self, hidden_states):
        hidden_states = self.conv(hidden_states)
        hidden_states = self.batch_norm(hidden_states)
        if self.activation is not None:
            hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states