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
def _create_causal_mask(self, key_length, query_length):
    causal_mask = torch.tril(torch.ones((self.max_positions, self.max_positions), dtype=torch.bool).view(1, 1, self.max_positions, self.max_positions))
    return causal_mask[:, :, key_length - query_length:key_length, :key_length]