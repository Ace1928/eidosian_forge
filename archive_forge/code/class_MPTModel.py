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
class MPTModel(nn.Module):

    def __init__(self, config: MPTConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        assert config.embedding_fraction == 1.0
        assert config.norm_type == 'low_precision_layernorm'
        self.wte = VocabParallelEmbedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([MPTBlock(config, linear_method) for _ in range(config.n_layers)])
        self.norm_f = nn.LayerNorm(config.d_model)
        if config.no_bias:
            for module in self.modules():
                if hasattr(module, 'bias') and isinstance(module.bias, nn.Parameter):
                    module.register_parameter('bias', None)

    def forward(self, input_ids: torch.Tensor, position_ids: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        for i in range(len(self.blocks)):
            block = self.blocks[i]
            hidden_states = block(position_ids, hidden_states, kv_caches[i], input_metadata)
        hidden_states = self.norm_f(hidden_states)
        return hidden_states