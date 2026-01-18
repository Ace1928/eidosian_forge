import math
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
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
from vllm.transformers_utils.configs.mpt import MPTConfig
class MPTBlock(nn.Module):

    def __init__(self, config: MPTConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        hidden_size = config.d_model
        self.norm_1 = nn.LayerNorm(hidden_size)
        self.attn = MPTAttention(config, linear_method)
        self.norm_2 = nn.LayerNorm(hidden_size)
        self.ffn = MPTMLP(config, linear_method)

    def forward(self, position_ids: torch.Tensor, hidden_states: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata) -> torch.Tensor:
        x = self.norm_1(hidden_states)
        x = self.attn(position_ids=position_ids, hidden_states=x, kv_cache=kv_cache, input_metadata=input_metadata)
        hidden_states = hidden_states + x
        x = self.norm_2(hidden_states)
        x = self.ffn(x)
        hidden_states = hidden_states + x
        return hidden_states