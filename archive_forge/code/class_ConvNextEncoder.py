from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...utils import (
from ...utils.backbone_utils import BackboneMixin
from .configuration_convnext import ConvNextConfig
class ConvNextEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.stages = nn.ModuleList()
        drop_path_rates = [x.tolist() for x in torch.linspace(0, config.drop_path_rate, sum(config.depths)).split(config.depths)]
        prev_chs = config.hidden_sizes[0]
        for i in range(config.num_stages):
            out_chs = config.hidden_sizes[i]
            stage = ConvNextStage(config, in_channels=prev_chs, out_channels=out_chs, stride=2 if i > 0 else 1, depth=config.depths[i], drop_path_rates=drop_path_rates[i])
            self.stages.append(stage)
            prev_chs = out_chs

    def forward(self, hidden_states: torch.FloatTensor, output_hidden_states: Optional[bool]=False, return_dict: Optional[bool]=True) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None
        for i, layer_module in enumerate(self.stages):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            hidden_states = layer_module(hidden_states)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states] if v is not None))
        return BaseModelOutputWithNoAttention(last_hidden_state=hidden_states, hidden_states=all_hidden_states)