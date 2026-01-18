from typing import List, Optional, Tuple
import torch
from torch import nn
from transformers import GPTBigCodeConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
class GPTBigMLP(nn.Module):

    def __init__(self, intermediate_size: int, config: GPTBigCodeConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        hidden_size = config.hidden_size
        self.c_fc = ColumnParallelLinear(hidden_size, intermediate_size, bias=True, linear_method=linear_method)
        self.c_proj = RowParallelLinear(intermediate_size, hidden_size, bias=True, linear_method=linear_method)
        quant_config = getattr(linear_method, 'quant_config', None)
        self.act = get_act_fn(config.activation_function, quant_config, intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states