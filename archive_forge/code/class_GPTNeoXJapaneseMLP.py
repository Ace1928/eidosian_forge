from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import logging
from .configuration_gpt_neox_japanese import GPTNeoXJapaneseConfig
class GPTNeoXJapaneseMLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        intermediate_size = int(config.hidden_size * config.intermediate_multiple_size)
        self.dense_h_to_4h = nn.Linear(config.hidden_size, intermediate_size, bias=False)
        self.dense_4h_to_h = nn.Linear(intermediate_size, config.hidden_size, bias=False)
        self.act = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        intermediate = self.dense_h_to_4h(hidden_states)
        intermediate = self.act(intermediate)
        output = self.dense_4h_to_h(intermediate)
        return output