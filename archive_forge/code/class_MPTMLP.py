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
class MPTMLP(nn.Module):

    def __init__(self, config: MPTConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        hidden_size = config.d_model
        expansion_ratio = config.expansion_ratio
        intermediate_size = expansion_ratio * hidden_size
        self.up_proj = ColumnParallelLinear(hidden_size, intermediate_size, bias=not config.no_bias, linear_method=linear_method)
        quant_config = getattr(linear_method, 'quant_config', None)
        self.act = get_act_fn('gelu', quant_config, intermediate_size)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, bias=not config.no_bias, linear_method=linear_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.up_proj(x)
        x = self.act(x)
        x, _ = self.down_proj(x)
        return x