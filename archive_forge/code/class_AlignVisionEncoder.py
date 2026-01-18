import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
from .configuration_align import AlignConfig, AlignTextConfig, AlignVisionConfig
class AlignVisionEncoder(nn.Module):
    """
    Forward propogates the embeddings through each vision encoder (EfficientNet) block.

    Args:
        config ([`AlignVisionConfig`]):
            Model configuration class.
    """

    def __init__(self, config: AlignVisionConfig):
        super().__init__()
        self.depth_coefficient = config.depth_coefficient

        def round_repeats(repeats):
            return int(math.ceil(self.depth_coefficient * repeats))
        num_base_blocks = len(config.in_channels)
        num_blocks = sum((round_repeats(n) for n in config.num_block_repeats))
        curr_block_num = 0
        blocks = []
        for i in range(num_base_blocks):
            in_dim = round_filters(config, config.in_channels[i])
            out_dim = round_filters(config, config.out_channels[i])
            stride = config.strides[i]
            kernel_size = config.kernel_sizes[i]
            expand_ratio = config.expand_ratios[i]
            for j in range(round_repeats(config.num_block_repeats[i])):
                id_skip = True if j == 0 else False
                stride = 1 if j > 0 else stride
                in_dim = out_dim if j > 0 else in_dim
                adjust_padding = False if curr_block_num in config.depthwise_padding else True
                drop_rate = config.drop_connect_rate * curr_block_num / num_blocks
                block = AlignVisionBlock(config=config, in_dim=in_dim, out_dim=out_dim, stride=stride, kernel_size=kernel_size, expand_ratio=expand_ratio, drop_rate=drop_rate, id_skip=id_skip, adjust_padding=adjust_padding)
                blocks.append(block)
                curr_block_num += 1
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states: torch.FloatTensor, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> BaseModelOutputWithPoolingAndNoAttention:
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        for block in self.blocks:
            hidden_states = block(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)