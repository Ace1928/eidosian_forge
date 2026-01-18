import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BackboneOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_nat import NatConfig
class NatPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, height, width, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        patch_size = config.patch_size
        num_channels, hidden_size = (config.num_channels, config.embed_dim)
        self.num_channels = num_channels
        if patch_size == 4:
            pass
        else:
            raise ValueError('Dinat only supports patch size of 4 at the moment.')
        self.projection = nn.Sequential(nn.Conv2d(self.num_channels, hidden_size // 2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)), nn.Conv2d(hidden_size // 2, hidden_size, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)))

    def forward(self, pixel_values: Optional[torch.FloatTensor]) -> torch.Tensor:
        _, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError('Make sure that the channel dimension of the pixel values match with the one set in the configuration.')
        embeddings = self.projection(pixel_values)
        embeddings = embeddings.permute(0, 2, 3, 1)
        return embeddings