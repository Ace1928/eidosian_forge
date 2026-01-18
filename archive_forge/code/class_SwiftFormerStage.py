import collections.abc
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2CLS
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_swiftformer import SwiftFormerConfig
class SwiftFormerStage(nn.Module):
    """
    A Swiftformer stage consisting of a series of `SwiftFormerConvEncoder` blocks and a final
    `SwiftFormerEncoderBlock`.

    Input: tensor in shape `[batch_size, channels, height, width]`

    Output: tensor in shape `[batch_size, channels, height, width]`
    """

    def __init__(self, config: SwiftFormerConfig, index: int) -> None:
        super().__init__()
        layer_depths = config.depths
        dim = config.embed_dims[index]
        depth = layer_depths[index]
        blocks = []
        for block_idx in range(depth):
            block_dpr = config.drop_path_rate * (block_idx + sum(layer_depths[:index])) / (sum(layer_depths) - 1)
            if depth - block_idx <= 1:
                blocks.append(SwiftFormerEncoderBlock(config, dim=dim, drop_path=block_dpr))
            else:
                blocks.append(SwiftFormerConvEncoder(config, dim=dim))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        for block in self.blocks:
            input = block(input)
        return input