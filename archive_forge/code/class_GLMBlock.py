from typing import List, Optional, Tuple
import torch
from torch import nn
from torch.nn import LayerNorm
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
from vllm.transformers_utils.configs import ChatGLMConfig
class GLMBlock(nn.Module):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, config, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.fp32_residual_connection = config.fp32_residual_connection
        layer_norm_func = RMSNorm if config.rmsnorm else LayerNorm
        self.input_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon)
        self.self_attention = GLMAttention(config, linear_method)
        self.hidden_dropout = config.hidden_dropout
        self.post_attention_layernorm = layer_norm_func(config.hidden_size, eps=config.layernorm_epsilon)
        self.mlp = GLMMLP(config, linear_method)

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor, kv_cache: KVCache, input_metadata: InputMetadata) -> torch.Tensor:
        layernorm_output = self.input_layernorm(hidden_states)
        attention_output = self.self_attention(hidden_states=layernorm_output, position_ids=position_ids, kv_cache=kv_cache, input_metadata=input_metadata)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        layernorm_input = residual + attention_output
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input
        output = self.mlp(layernorm_output) + residual
        return output