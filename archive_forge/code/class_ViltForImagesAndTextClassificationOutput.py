import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import (
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_vilt import ViltConfig
@dataclass
class ViltForImagesAndTextClassificationOutput(ModelOutput):
    """
    Class for outputs of [`ViltForImagesAndTextClassification`].

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (`List[tuple(torch.FloatTensor)]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            List of tuples of `torch.FloatTensor` (one for each image-text pair, each tuple containing the output of
            the embeddings + one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`List[tuple(torch.FloatTensor)]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            List of tuples of `torch.FloatTensor` (one for each image-text pair, each tuple containing the attention
            weights of shape `(batch_size, num_heads, sequence_length, sequence_length)`. Attentions weights after the
            attention softmax, used to compute the weighted average in the self-attention heads.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[List[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[List[Tuple[torch.FloatTensor]]] = None