import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
from torch import nn
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import ModelOutput, add_start_docstrings, logging, replace_return_docstrings
from .configuration_fastspeech2_conformer import (
@dataclass
class FastSpeech2ConformerWithHifiGanOutput(FastSpeech2ConformerModelOutput):
    """
    Output type of [`FastSpeech2ConformerWithHifiGan`].

    Args:
        waveform (`torch.FloatTensor` of shape `(batch_size, audio_length)`):
            Speech output as a result of passing the predicted mel spectrogram through the vocoder.
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Spectrogram generation loss.
        spectrogram (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_bins)`):
            The predicted spectrogram.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the encoder at the output of each layer plus the initial embedding outputs.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the decoder at the output of each layer plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights of the decoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        duration_outputs (`torch.LongTensor` of shape `(batch_size, max_text_length + 1)`, *optional*):
            Outputs of the duration predictor.
        pitch_outputs (`torch.FloatTensor` of shape `(batch_size, max_text_length + 1, 1)`, *optional*):
            Outputs of the pitch predictor.
        energy_outputs (`torch.FloatTensor` of shape `(batch_size, max_text_length + 1, 1)`, *optional*):
            Outputs of the energy predictor.
    """
    waveform: torch.FloatTensor = None