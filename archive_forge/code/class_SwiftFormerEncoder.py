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
class SwiftFormerEncoder(nn.Module):

    def __init__(self, config: SwiftFormerConfig) -> None:
        super().__init__()
        self.config = config
        embed_dims = config.embed_dims
        downsamples = config.downsamples
        layer_depths = config.depths
        network = []
        for i in range(len(layer_depths)):
            stage = SwiftFormerStage(config=config, index=i)
            network.append(stage)
            if i >= len(layer_depths) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                network.append(SwiftFormerEmbeddings(config, index=i))
        self.network = nn.ModuleList(network)
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, output_hidden_states: Optional[bool]=None, return_dict: Optional[bool]=None) -> Union[tuple, BaseModelOutputWithNoAttention]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        all_hidden_states = (hidden_states,) if output_hidden_states else None
        for block in self.network:
            hidden_states = block(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)