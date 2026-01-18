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
class PerceiverMultimodalPostprocessor(nn.Module):
    """
    Multimodal postprocessing for Perceiver. Can be used to combine modality-specific postprocessors into a single
    postprocessor.

    Args:
          modalities (`Mapping[str, PostprocessorType]`):
            Dictionary mapping modality name to postprocessor class for that modality.
          input_is_dict (`bool`, *optional*, defaults to `False`):
            If True, input is assumed to be dictionary structured, and outputs keep the same dictionary shape. If
            False, input is a tensor which is sliced up during postprocessing by *modality_sizes*.
    """

    def __init__(self, modalities: Mapping[str, PostprocessorType], input_is_dict: bool=False):
        super().__init__()
        self.modalities = nn.ModuleDict(modalities)
        self.input_is_dict = input_is_dict

    def forward(self, inputs: torch.Tensor, pos: Optional[torch.Tensor]=None, modality_sizes=None) -> Mapping[str, torch.Tensor]:
        if not self.input_is_dict:
            if modality_sizes is None:
                raise ValueError('Modality sizes should be specified if input is not a dictionary.')
            inputs = restructure(modality_sizes=modality_sizes, inputs=inputs)
        outputs = {modality: postprocessor(inputs[modality], pos=pos, modality_sizes=None) for modality, postprocessor in self.modalities.items()}
        return outputs