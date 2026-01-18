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
class EfficientFormerIntermediateStage(nn.Module):

    def __init__(self, config: EfficientFormerConfig, index: int):
        super().__init__()
        self.meta4D_layers = EfficientFormerMeta4DLayers(config, index)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor]:
        hidden_states = self.meta4D_layers(hidden_states)
        return hidden_states