import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_rwkv import RwkvConfig
class RwkvSelfAttention(nn.Module):

    def __init__(self, config, layer_id=0):
        super().__init__()
        self.config = config
        kernel_loaded = rwkv_cuda_kernel is not None and rwkv_cuda_kernel.max_seq_length == config.context_length
        if is_ninja_available() and is_torch_cuda_available() and (not kernel_loaded):
            try:
                load_wkv_cuda_kernel(config.context_length)
            except Exception:
                logger.info('Could not load the custom CUDA kernel for RWKV attention.')
        self.layer_id = layer_id
        hidden_size = config.hidden_size
        attention_hidden_size = config.attention_hidden_size if config.attention_hidden_size is not None else hidden_size
        self.attention_hidden_size = attention_hidden_size
        self.time_decay = nn.Parameter(torch.empty(attention_hidden_size))
        self.time_first = nn.Parameter(torch.empty(attention_hidden_size))
        self.time_mix_key = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_value = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_mix_receptance = nn.Parameter(torch.empty(1, 1, hidden_size))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.receptance = nn.Linear(hidden_size, attention_hidden_size, bias=False)
        self.output = nn.Linear(attention_hidden_size, hidden_size, bias=False)

    def extract_key_value(self, hidden, state=None):
        if hidden.size(1) == 1 and state is not None:
            shifted = state[1][:, :, self.layer_id]
        else:
            shifted = self.time_shift(hidden)
            if state is not None:
                shifted[:, 0] = state[1][:, :, self.layer_id]
        key = hidden * self.time_mix_key + shifted * (1 - self.time_mix_key)
        value = hidden * self.time_mix_value + shifted * (1 - self.time_mix_value)
        receptance = hidden * self.time_mix_receptance + shifted * (1 - self.time_mix_receptance)
        key = self.key(key)
        value = self.value(value)
        receptance = torch.sigmoid(self.receptance(receptance))
        if state is not None:
            state[1][:, :, self.layer_id] = hidden[:, -1]
        return (receptance, key, value, state)

    def forward(self, hidden, state=None, use_cache=False):
        receptance, key, value, state = self.extract_key_value(hidden, state=state)
        layer_state = tuple((s[:, :, self.layer_id] for s in state[2:])) if state is not None else None
        rwkv, layer_state = rwkv_linear_attention(self.time_decay, self.time_first, key, value, state=layer_state, return_state=use_cache)
        if layer_state is not None:
            state[2][:, :, self.layer_id] = layer_state[0]
            state[3][:, :, self.layer_id] = layer_state[1]
            state[4][:, :, self.layer_id] = layer_state[2]
        return (self.output(receptance * rwkv), state)