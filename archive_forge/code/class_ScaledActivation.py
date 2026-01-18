import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm._C import ops
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.parallel_utils.parallel_state import (
from vllm.model_executor.parallel_utils.utils import divide
from vllm.model_executor.utils import set_weight_attrs
class ScaledActivation(nn.Module):
    """An activation function with post-scale parameters.

    This is used for some quantization methods like AWQ.
    """

    def __init__(self, act_module: nn.Module, intermediate_size: int, input_is_parallel: bool=True, params_dtype: Optional[torch.dtype]=None):
        super().__init__()
        self.act = act_module
        self.input_is_parallel = input_is_parallel
        if input_is_parallel:
            tp_size = get_tensor_model_parallel_world_size()
            intermediate_size_per_partition = divide(intermediate_size, tp_size)
        else:
            intermediate_size_per_partition = intermediate_size
        if params_dtype is None:
            params_dtype = torch.get_default_dtype()
        self.scales = nn.Parameter(torch.empty(intermediate_size_per_partition, dtype=params_dtype))
        set_weight_attrs(self.scales, {'weight_loader': self.weight_loader})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x) / self.scales

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if self.input_is_parallel:
            tp_rank = get_tensor_model_parallel_rank()
            shard_size = param_data.shape[0]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)