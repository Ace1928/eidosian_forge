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
class GLMMLP(nn.Module):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, config, linear_method: Optional[LinearMethodBase]=None):
        super().__init__()
        self.add_bias = config.add_bias_linear
        self.dense_h_to_4h = MergedColumnParallelLinear(config.hidden_size, [config.ffn_hidden_size] * 2, bias=config.add_bias_linear, linear_method=linear_method)
        self.activation_func = SiluAndMul()
        self.dense_4h_to_h = RowParallelLinear(config.ffn_hidden_size, config.hidden_size, bias=config.add_bias_linear, linear_method=linear_method)

    def forward(self, hidden_states):
        intermediate_parallel, _ = self.dense_h_to_4h(hidden_states)
        intermediate_parallel = self.activation_func(intermediate_parallel)
        output, _ = self.dense_4h_to_h(intermediate_parallel)
        return output