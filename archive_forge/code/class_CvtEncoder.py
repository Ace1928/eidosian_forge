import collections.abc
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ...modeling_outputs import ImageClassifierOutputWithNoAttention, ModelOutput
from ...modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import logging
from .configuration_cvt import CvtConfig
class CvtEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stages = nn.ModuleList([])
        for stage_idx in range(len(config.depth)):
            self.stages.append(CvtStage(config, stage_idx))

    def forward(self, pixel_values, output_hidden_states=False, return_dict=True):
        all_hidden_states = () if output_hidden_states else None
        hidden_state = pixel_values
        cls_token = None
        for _, stage_module in enumerate(self.stages):
            hidden_state, cls_token = stage_module(hidden_state)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)
        if not return_dict:
            return tuple((v for v in [hidden_state, cls_token, all_hidden_states] if v is not None))
        return BaseModelOutputWithCLSToken(last_hidden_state=hidden_state, cls_token_value=cls_token, hidden_states=all_hidden_states)