from typing import List, Optional, Tuple
import torch
from torch import nn
from transformers import MixtralConfig
from vllm.config import LoRAConfig
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.attention import PagedAttention
from vllm.model_executor.layers.fused_moe import fused_moe
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (LinearMethodBase,
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
from vllm.model_executor.parallel_utils.communication_op import (
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.weight_utils import (default_weight_loader,
from vllm.sequence import SamplerOutput
class MixtralMoE(nn.Module):
    """A tensor-parallel MoE implementation for Mixtral that shards each expert
    across all ranks.

    Each expert's weights are sharded across all ranks and a fused MoE
    kernel is used for the forward pass, and finally we reduce the outputs
    across ranks.
    """

    def __init__(self, num_experts: int, top_k: int, hidden_size: int, intermediate_size: int, params_dtype: Optional[torch.dtype]=None, tp_size: Optional[int]=None):
        super().__init__()
        self.tp_size = tp_size or get_tensor_model_parallel_world_size()
        self.num_total_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size // self.tp_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.params_dtype = params_dtype
        self.gate = ReplicatedLinear(self.hidden_size, self.num_total_experts, bias=False, params_dtype=self.params_dtype, linear_method=None)
        self.ws = nn.Parameter(torch.empty(self.num_total_experts, 2 * self.intermediate_size, self.hidden_size, device='cuda', dtype=self.params_dtype))
        self.w2s = nn.Parameter(torch.empty(self.num_total_experts, self.hidden_size, self.intermediate_size, device='cuda', dtype=self.params_dtype))
        set_weight_attrs(self.ws, {'weight_loader': self.weight_loader})
        set_weight_attrs(self.w2s, {'weight_loader': self.weight_loader})

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str, expert_id: int):
        tp_rank = get_tensor_model_parallel_rank()
        param_data = param.data
        shard_size = self.intermediate_size
        shard = slice(tp_rank * shard_size, (tp_rank + 1) * shard_size)
        if weight_name.endswith('w1.weight'):
            param_data[expert_id, 0:shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith('w3.weight'):
            param_data[expert_id, shard_size:2 * shard_size, :] = loaded_weight[shard, :]
        if weight_name.endswith('w2.weight'):
            param_data[expert_id, :, :] = loaded_weight[:, shard]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits, _ = self.gate(hidden_states)
        final_hidden_states = fused_moe(hidden_states, self.ws, self.w2s, router_logits, self.top_k, renormalize=True, inplace=True)
        if self.tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
        return final_hidden_states.view(batch_size, sequence_length, hidden_size)