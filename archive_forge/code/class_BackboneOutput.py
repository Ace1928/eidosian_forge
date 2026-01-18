import warnings
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from .utils import ModelOutput
@dataclass
class BackboneOutput(ModelOutput):
    """
    Base class for outputs of backbones.

    Args:
        feature_maps (`tuple(torch.FloatTensor)` of shape `(batch_size, num_channels, height, width)`):
            Feature maps of the stages.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, num_channels, height, width)`,
            depending on the backbone.

            Hidden-states of the model at the output of each stage plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Only applicable if the backbone uses attention.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    feature_maps: Tuple[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None