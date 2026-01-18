import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_perceiver import PerceiverConfig
class PerceiverBasicVideoAutoencodingDecoder(PerceiverAbstractDecoder):
    """
    Cross-attention based video-autoencoding decoder. Light-weight wrapper of [*PerceiverBasicDecoder*] with video
    reshaping logic.

    Args:
        config ([*PerceiverConfig*]):
            Model configuration.
        output_shape (`List[int]`):
            Shape of the output as (batch_size, num_frames, height, width), excluding the channel dimension.
        position_encoding_type (`str`):
            The type of position encoding to use. Can be either "trainable", "fourier", or "none".
    """

    def __init__(self, config: PerceiverConfig, output_shape: List[int], position_encoding_type: str, **decoder_kwargs) -> None:
        super().__init__()
        if len(output_shape) != 4:
            raise ValueError(f'Expected rank 4 output_shape, got {output_shape}.')
        self.output_shape = output_shape
        self.output_num_channels = decoder_kwargs['output_num_channels']
        self.decoder = PerceiverBasicDecoder(config, output_index_dims=self.output_shape[1:4], position_encoding_type=position_encoding_type, **decoder_kwargs)

    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        return self.decoder.decoder_query(inputs, modality_sizes=modality_sizes, inputs_without_pos=inputs_without_pos, subsampled_points=subsampled_points)

    def forward(self, query: torch.Tensor, z: torch.FloatTensor, query_mask: Optional[torch.FloatTensor]=None) -> PerceiverDecoderOutput:
        decoder_outputs = self.decoder(query, z)
        logits = decoder_outputs.logits
        logits = torch.reshape(logits, self.output_shape + [logits.shape[-1]])
        return PerceiverDecoderOutput(logits=logits, cross_attentions=decoder_outputs.cross_attentions)