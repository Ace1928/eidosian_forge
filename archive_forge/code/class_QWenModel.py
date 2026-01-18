from typing import Any, Dict, List, Optional, Tuple
import torch
from torch import nn
from transformers import PretrainedConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
class QWenModel(nn.Module):

    def __init__(self, config: PretrainedConfig, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.wte = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.h = nn.ModuleList([QWenBlock(config, linear_method) for _ in range(config.num_hidden_layers)])
        self.ln_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata) -> torch.Tensor:
        hidden_states = self.wte(input_ids)
        residual = None
        for i in range(len(self.h)):
            layer = self.h[i]
            hidden_states, residual = layer(positions, hidden_states, kv_caches[i], input_metadata, residual)
        hidden_states, _ = self.ln_f(hidden_states, residual)
        return hidden_states