import math
from dataclasses import dataclass
from numbers import Number
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor, nn
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import load_backbone
from ..detr import DetrConfig
from .configuration_maskformer import MaskFormerConfig
from .configuration_maskformer_swin import MaskFormerSwinConfig
@dataclass
class MaskFormerModelOutput(ModelOutput):
    """
    Class for outputs of [`MaskFormerModel`]. This class returns all the needed hidden states to compute the logits.

    Args:
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the encoder model (backbone).
        pixel_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Last hidden states (final feature map) of the last stage of the pixel decoder model (FPN).
        transformer_decoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Last hidden states (final feature map) of the last stage of the transformer decoder model.
        encoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the encoder
            model at the output of each stage.
        pixel_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, num_channels, height, width)`. Hidden-states (also called feature maps) of the pixel
            decoder model at the output of each stage.
        transformer_decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each stage) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states (also called feature maps) of the
            transformer decoder at the output of each stage.
        hidden_states `tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` containing `encoder_hidden_states`, `pixel_decoder_hidden_states` and
            `decoder_hidden_states`
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights from Detr's decoder after the attention softmax, used to compute the
            weighted average in the self-attention heads.
    """
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    pixel_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    transformer_decoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pixel_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    transformer_decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None