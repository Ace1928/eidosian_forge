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
class Starcoder2Model(nn.Module):

    def __init__(self, config: Starcoder2Config, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Starcoder2DecoderLayer(config, linear_method=linear_method) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.norm_epsilon)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor, kv_caches: List[KVCache], input_metadata: InputMetadata) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(positions, hidden_states, kv_caches[i], input_metadata)
        hidden_states = self.norm(hidden_states)
        return hidden_states