from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.linear import (
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (
from vllm.sequence import SamplerOutput
from hf_olmo import OLMoConfig
class OlmoMLP(nn.Module):
    """
    This is the MLP block where the output is computed as ``MLP(LN(x))`` in ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, config: OLMoConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        self.ff_norm = nn.LayerNorm(config.d_model, elementwise_affine=False, bias=False)
        self.ff_proj = ColumnParallelLinear(config.d_model, self.hidden_size, bias=config.include_bias, linear_method=linear_method)
        self.act = SwiGLU()
        assert self.act.output_multiplier * self.hidden_size % 1 == 0
        self.ff_out = RowParallelLinear(int(self.act.output_multiplier * self.hidden_size), config.d_model, bias=config.include_bias, linear_method=linear_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        og_x = x
        x = self.ff_norm(x)
        x, _ = self.ff_proj(x)
        x = self.act(x)
        x, _ = self.ff_out(x)
        x = og_x + x
        return x