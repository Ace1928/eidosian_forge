import itertools
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_efficientformer import EfficientFormerConfig
class EfficientFormerLastStage(nn.Module):

    def __init__(self, config: EfficientFormerConfig):
        super().__init__()
        self.meta4D_layers = EfficientFormerMeta4DLayers(config, -1)
        self.flat = EfficientFormerFlat()
        self.meta3D_layers = EfficientFormerMeta3DLayers(config)

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool=False) -> Tuple[torch.Tensor]:
        hidden_states = self.meta4D_layers(hidden_states)
        hidden_states = self.flat(hidden_states)
        hidden_states = self.meta3D_layers(hidden_states, output_attentions)
        return hidden_states