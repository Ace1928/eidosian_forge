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
class EfficientFormerEncoder(nn.Module):

    def __init__(self, config: EfficientFormerConfig):
        super().__init__()
        self.config = config
        num_intermediate_stages = len(config.depths) - 1
        downsamples = [config.downsamples[i] or config.hidden_sizes[i] != config.hidden_sizes[i + 1] for i in range(num_intermediate_stages)]
        intermediate_stages = []
        for i in range(num_intermediate_stages):
            intermediate_stages.append(EfficientFormerIntermediateStage(config, i))
            if downsamples[i]:
                intermediate_stages.append(EfficientFormerPatchEmbeddings(config, config.hidden_sizes[i], config.hidden_sizes[i + 1]))
        self.intermediate_stages = nn.ModuleList(intermediate_stages)
        self.last_stage = EfficientFormerLastStage(config)

    def forward(self, hidden_states: torch.Tensor, output_hidden_states: bool=False, output_attentions: bool=False, return_dict: bool=True) -> BaseModelOutput:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        for layer_module in self.intermediate_stages:
            hidden_states = layer_module(hidden_states)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
        layer_output = self.last_stage(hidden_states, output_attentions=output_attentions)
        if output_attentions:
            all_self_attentions = all_self_attentions + layer_output[1:]
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (layer_output[0],)
        if not return_dict:
            return tuple((v for v in [layer_output[0], all_hidden_states, all_self_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=layer_output[0], hidden_states=all_hidden_states, attentions=all_self_attentions)