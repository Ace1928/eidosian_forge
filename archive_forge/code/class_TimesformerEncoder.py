import collections
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_timesformer import TimesformerConfig
class TimesformerEncoder(nn.Module):

    def __init__(self, config: TimesformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([TimesformerLayer(config, ind) for ind in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, output_attentions: bool=False, output_hidden_states: bool=False, return_dict: bool=True) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(layer_module.__call__, hidden_states, output_attentions)
            else:
                layer_outputs = layer_module(hidden_states, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)