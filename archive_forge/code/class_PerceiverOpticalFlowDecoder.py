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
class PerceiverOpticalFlowDecoder(PerceiverAbstractDecoder):
    """Cross-attention based optical flow decoder."""

    def __init__(self, config, output_image_shape, output_num_channels=2, rescale_factor=100.0, **decoder_kwargs):
        super().__init__()
        self.output_image_shape = output_image_shape
        self.output_num_channels = output_num_channels
        self.rescale_factor = rescale_factor
        self.decoder = PerceiverBasicDecoder(config, output_num_channels=output_num_channels, **decoder_kwargs)

    @property
    def num_query_channels(self) -> int:
        return self.decoder.num_query_channels

    def decoder_query(self, inputs, modality_sizes=None, inputs_without_pos=None, subsampled_points=None):
        if subsampled_points is not None:
            raise ValueError("FlowDecoder doesn't support subsampling yet.")
        return inputs

    def forward(self, query: torch.Tensor, z: torch.FloatTensor, query_mask: Optional[torch.FloatTensor]=None, output_attentions: Optional[bool]=False) -> PerceiverDecoderOutput:
        decoder_outputs = self.decoder(query, z, output_attentions=output_attentions)
        preds = decoder_outputs.logits
        preds /= self.rescale_factor
        preds = preds.reshape([preds.shape[0]] + list(self.output_image_shape) + [preds.shape[-1]])
        return PerceiverDecoderOutput(logits=preds, cross_attentions=decoder_outputs.cross_attentions)