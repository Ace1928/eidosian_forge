from typing import List, Optional, Tuple
import torch
from torch import nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.parallel_state import get_tensor_model_parallel_world_size
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
class Starcoder2MLP(nn.Module):

    def __init__(self, config: Starcoder2Config, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.c_fc = ColumnParallelLinear(config.hidden_size, config.intermediate_size, bias=config.use_bias, linear_method=linear_method)
        self.c_proj = RowParallelLinear(config.intermediate_size, config.hidden_size, bias=config.use_bias, linear_method=linear_method)
        self.act = get_act_fn(config.hidden_act, intermediate_size=config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states, _ = self.c_proj(hidden_states)
        return hidden_states