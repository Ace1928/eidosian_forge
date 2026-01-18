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
class SpeechT5SpeechDecoderPrenet(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([nn.Linear(config.num_mel_bins if i == 0 else config.speech_decoder_prenet_units, config.speech_decoder_prenet_units) for i in range(config.speech_decoder_prenet_layers)])
        self.final_layer = nn.Linear(config.speech_decoder_prenet_units, config.hidden_size)
        self.encode_positions = SpeechT5ScaledPositionalEncoding(config.positional_dropout, config.hidden_size, config.max_speech_positions)
        self.speaker_embeds_layer = nn.Linear(config.speaker_embedding_dim + config.hidden_size, config.hidden_size)

    def _consistent_dropout(self, inputs_embeds, p):
        mask = torch.bernoulli(inputs_embeds[0], p=p)
        all_masks = mask.unsqueeze(0).repeat(inputs_embeds.size(0), 1, 1)
        return torch.where(all_masks == 1, inputs_embeds, 0) * 1 / (1 - p)

    def forward(self, input_values: torch.Tensor, speaker_embeddings: Optional[torch.Tensor]=None):
        inputs_embeds = input_values
        for layer in self.layers:
            inputs_embeds = nn.functional.relu(layer(inputs_embeds))
            inputs_embeds = self._consistent_dropout(inputs_embeds, self.config.speech_decoder_prenet_dropout)
        inputs_embeds = self.final_layer(inputs_embeds)
        inputs_embeds = self.encode_positions(inputs_embeds)
        if speaker_embeddings is not None:
            speaker_embeddings = nn.functional.normalize(speaker_embeddings)
            speaker_embeddings = speaker_embeddings.unsqueeze(1).expand(-1, inputs_embeds.size(1), -1)
            inputs_embeds = torch.cat([inputs_embeds, speaker_embeddings], dim=-1)
            inputs_embeds = nn.functional.relu(self.speaker_embeds_layer(inputs_embeds))
        return inputs_embeds